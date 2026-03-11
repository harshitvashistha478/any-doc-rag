"""
RAG utility — all five high-priority features wired together:

  Feature 1 & 2 — Per-user document isolation via per-user Milvus collections.
                  Every user's chunks live in collection `user_<id>`, so no
                  cross-user leakage is possible at the DB level.

  Feature 3     — Conversation memory.  generate_answer() accepts chat_history
                  (list of {role, content} dicts) and injects it into the
                  prompt so the LLM can handle follow-up questions.

  Feature 4     — Cross-encoder re-ranking.  We fetch 3× the requested k from
                  Milvus (cheap vector search), then re-score every candidate
                  with a CrossEncoder (accurate but slower) and keep the top k.

  Feature 5     — Indexing status tracking.  index_file_task() is an async
                  background coroutine that writes PROCESSING → INDEXED/FAILED
                  back to Postgres so callers can poll for completion.
"""

from __future__ import annotations

import logging
import os
from typing import List, Tuple, Dict

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import CrossEncoder
from sqlalchemy import update

load_dotenv()

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────
MILVUS_URI           = os.getenv("MILVUS_URI", "http://localhost:19530")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL_NAME       = os.getenv("LLM_MODEL_NAME", "llama3-8b-8192")
RERANKER_MODEL_NAME  = os.getenv("RERANKER_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")

# ── Shared model instances (loaded once at startup) ───────────────────────────
llm        = ChatGroq(model=LLM_MODEL_NAME, temperature=0.7)
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
reranker   = CrossEncoder(RERANKER_MODEL_NAME)   # Feature 4


# ─────────────────────────────────────────────────────────────────────────────
# Feature 1 & 2 — Per-user Milvus collection helpers
# ─────────────────────────────────────────────────────────────────────────────

def _collection_name(user_id: int) -> str:
    """Each user gets their own isolated Milvus collection."""
    return f"user_{user_id}"


def _get_vector_store(user_id: int) -> Milvus:
    """Return a handle to the user's Milvus collection (read-only, no docs added)."""
    return Milvus(
        embedding_function=embeddings,
        collection_name=_collection_name(user_id),
        connection_args={"uri": MILVUS_URI},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Document loading & splitting
# ─────────────────────────────────────────────────────────────────────────────

def _load_single_file(file_path: str) -> List[Document]:
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load()
    logger.info("Loaded %d document(s) from %s", len(docs), file_path)
    return docs


def _split_docs(
    data: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(data)
    logger.info("Split into %d chunks", len(chunks))
    return chunks


def _save_to_user_collection(
    chunks: List[Document],
    user_id: int,
    file_id: str,
    file_name: str,
) -> None:
    """
    Store chunks in the user's private Milvus collection.
    Each chunk carries metadata so we can cite sources and
    delete by file_id later.
    """
    # Inject source metadata into every chunk
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "user_id":   user_id,
            "file_id":   file_id,
            "file_name": file_name,
            "chunk_idx": i,
        })

    Milvus.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=_collection_name(user_id),   # Feature 1 & 2
        connection_args={"uri": MILVUS_URI},
        drop_old=False,   # append — preserve existing user docs
    )
    logger.info(
        "Saved %d chunks to collection '%s' for file_id=%s",
        len(chunks), _collection_name(user_id), file_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature 4 — Re-ranking
# ─────────────────────────────────────────────────────────────────────────────

def _rerank(query: str, docs: List[Document], top_k: int) -> List[Document]:
    """
    Score every candidate doc with the CrossEncoder, then return the
    top_k by descending relevance score.
    """
    if not docs:
        return []

    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)                    # shape: (len(docs),)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top    = [doc for _, doc in ranked[:top_k]]

    logger.info(
        "Re-ranked %d candidates → kept top %d  (best score: %.4f)",
        len(docs), len(top), ranked[0][0] if ranked else 0,
    )
    return top


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5 — Indexing status tracking (async background task)
# ─────────────────────────────────────────────────────────────────────────────

async def index_file_task(
    file_path: str,
    user_id: int,
    file_id: str,
    file_name: str,
) -> None:
    """
    Full indexing pipeline run as an async FastAPI BackgroundTask.
    Opens its own DB session so it can update the file's indexing_status
    independently of the HTTP request that spawned it.

    Status transitions:
        PENDING → PROCESSING → INDEXED
                            ↘ FAILED  (on any exception)
    """
    # Import here to avoid circular imports at module load time
    from src.database.config import AsyncSessionLocal
    from src.models.files import FileInputModel, IndexingStatus

    async with AsyncSessionLocal() as db:
        # ── Mark as processing ───────────────────────────────────────────────
        await db.execute(
            update(FileInputModel)
            .where(FileInputModel.file_id == file_id)
            .values(indexing_status=IndexingStatus.PROCESSING)
        )
        await db.commit()

        try:
            docs   = _load_single_file(file_path)
            chunks = _split_docs(docs)
            _save_to_user_collection(chunks, user_id, file_id, file_name)

            # ── Mark as indexed ──────────────────────────────────────────────
            await db.execute(
                update(FileInputModel)
                .where(FileInputModel.file_id == file_id)
                .values(
                    indexing_status=IndexingStatus.INDEXED,
                    indexing_error=None,
                )
            )
            await db.commit()
            logger.info("Indexing complete for file_id=%s", file_id)

        except Exception as exc:
            # ── Mark as failed ───────────────────────────────────────────────
            logger.exception("Indexing failed for file_id=%s", file_id)
            await db.execute(
                update(FileInputModel)
                .where(FileInputModel.file_id == file_id)
                .values(
                    indexing_status=IndexingStatus.FAILED,
                    indexing_error=str(exc),
                )
            )
            await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Public retrieval helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_relevant_docs(
    query: str,
    user_id: int,
    top_k: int = 4,
    fetch_k: int = 12,    # fetch 3× more candidates for the re-ranker
) -> List[Document]:
    """
    Retrieve documents from the user's private collection (Feature 1 & 2),
    then re-rank them with the CrossEncoder (Feature 4).
    """
    vs        = _get_vector_store(user_id)
    retriever = vs.as_retriever(search_kwargs={"k": fetch_k})
    candidates = retriever.invoke(query)
    return _rerank(query, candidates, top_k)


def get_relevant_docs_with_score(
    query: str,
    user_id: int,
    top_k: int = 4,
    fetch_k: int = 12,
) -> List[Tuple[Document, float]]:
    """Same as above but also returns the re-ranker score for each chunk."""
    vs         = _get_vector_store(user_id)
    candidates = vs.similarity_search(query, k=fetch_k)
    if not candidates:
        return []

    pairs  = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [(doc, float(score)) for score, doc in ranked[:top_k]]


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — Conversation-aware answer generation
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    user_id: int,
    chat_history: List[Dict[str, str]] | None = None,
) -> Dict:
    """
    Full RAG pipeline with conversation memory and source citations.

    Args:
        query:        Current user question.
        user_id:      Used to scope retrieval to the user's own documents.
        chat_history: List of {"role": "user"|"assistant", "content": "..."}
                      dicts representing previous turns in this session.

    Returns:
        {
            "answer":  str,
            "sources": [{"file_name": str, "file_id": str, "chunk_idx": int}, ...]
        }
    """
    # ── Retrieve & re-rank ────────────────────────────────────────────────────
    docs_with_scores = get_relevant_docs_with_score(query, user_id)

    if not docs_with_scores:
        return {
            "answer": (
                "I could not find any relevant information in your documents "
                "to answer this question."
            ),
            "sources": [],
        }

    # ── Build context block with inline source tags ───────────────────────────
    context_parts = []
    sources       = []
    seen_sources  = set()

    for doc, score in docs_with_scores:
        meta       = doc.metadata
        file_name  = meta.get("file_name", "unknown")
        file_id    = meta.get("file_id", "")
        chunk_idx  = meta.get("chunk_idx", 0)

        context_parts.append(
            f"[Source: {file_name}, chunk {chunk_idx}, score {score:.3f}]\n"
            f"{doc.page_content}"
        )

        source_key = f"{file_id}:{chunk_idx}"
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "file_name": file_name,
                "file_id":   file_id,
                "chunk_idx": chunk_idx,
            })

    context = "\n\n---\n\n".join(context_parts)

    # ── Build history block ───────────────────────────────────────────────────
    # Feature 3: inject prior turns so the LLM can handle follow-up questions
    history_block = ""
    if chat_history:
        turns = []
        for turn in chat_history[-6:]:   # cap at last 6 turns to stay within context window
            role    = turn.get("role", "user").capitalize()
            content = turn.get("content", "")
            turns.append(f"{role}: {content}")
        history_block = "\n\n<Conversation History>\n" + "\n".join(turns) + "\n</Conversation History>"

    # ── Compose final prompt ──────────────────────────────────────────────────
    prompt = f"""You are a helpful assistant that answers questions strictly using the provided context.
If the answer is not found in the context, say so clearly — do not invent information.
{history_block}

<Context>
{context}
</Context>

Question: {query}

Answer:"""

    response = llm.invoke(prompt)

    return {
        "answer":  response.content,
        "sources": sources,
    }