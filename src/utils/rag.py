"""
RAG utility layer.

High-priority features already present
───────────────────────────────────────
  Features 1 & 2  Per-user Milvus collections  (user_{id})
  Feature  3      Conversation memory          (chat_history param)
  Feature  4      Cross-encoder re-ranking
  Feature  5      Async indexing-status updates

Medium-priority additions in this file
───────────────────────────────────────
  Task 6   stream_answer()   — async generator that yields LLM tokens one by
                               one so the caller can push them as SSE.

  Task 8   delete_file_vectors() — removes every Milvus chunk that belongs to
                               a specific (user_id, file_id) pair, called from
                               the DELETE /files/{file_id} endpoint so nothing
                               is left behind in the vector store.
"""

from __future__ import annotations

import logging
import os
from typing import AsyncIterator, Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_milvus import Milvus
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import Collection, connections, utility
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
reranker   = CrossEncoder(RERANKER_MODEL_NAME)


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers — collection naming & vector store handles
# ─────────────────────────────────────────────────────────────────────────────

def _collection_name(user_id: int) -> str:
    return f"user_{user_id}"


def _get_vector_store(user_id: int) -> Milvus:
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
    docs   = loader.load()
    logger.info("Loaded %d doc(s) from %s", len(docs), file_path)
    return docs


def _split_docs(
    data: List[Document],
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
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
    """Embed chunks and store them in the user's private Milvus collection."""
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
        collection_name=_collection_name(user_id),
        connection_args={"uri": MILVUS_URI},
        drop_old=False,
    )
    logger.info(
        "Saved %d chunks to collection '%s' (file_id=%s)",
        len(chunks), _collection_name(user_id), file_id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 8 — Delete vectors from Milvus when a file is deleted
# ─────────────────────────────────────────────────────────────────────────────

def delete_file_vectors(user_id: int, file_id: str) -> int:
    """
    Remove every chunk that belongs to `file_id` from the user's Milvus
    collection.  Returns the number of entities deleted.

    Strategy
    ────────
    We use pymilvus directly because langchain-milvus' high-level API only
    supports deletion by primary key.  Milvus 2.3+ supports scalar-field
    expression deletes, which is exactly what we need here.

    If the user's collection doesn't exist yet (edge case: file was uploaded
    but indexing never ran), we just return 0 safely.
    """
    col_name = _collection_name(user_id)

    try:
        connections.connect(alias="default", uri=MILVUS_URI)

        if not utility.has_collection(col_name):
            logger.info("Collection '%s' does not exist — nothing to delete.", col_name)
            return 0

        collection = Collection(col_name)
        collection.load()

        # Expression filter: match the file_id metadata field
        expr   = f'file_id == "{file_id}"'
        result = collection.delete(expr=expr)

        deleted = result.delete_count
        logger.info(
            "Deleted %d vector(s) from collection '%s' for file_id='%s'",
            deleted, col_name, file_id,
        )
        return deleted

    except Exception as exc:
        # Non-fatal: log and continue — the file record will still be removed
        # from Postgres even if vector cleanup fails.
        logger.error(
            "Failed to delete vectors for file_id='%s' in collection '%s': %s",
            file_id, col_name, exc,
        )
        return 0


# ─────────────────────────────────────────────────────────────────────────────
# Re-ranking (Feature 4)
# ─────────────────────────────────────────────────────────────────────────────

def _rerank(query: str, docs: List[Document], top_k: int) -> List[Document]:
    if not docs:
        return []
    pairs  = [(query, doc.page_content) for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    top    = [doc for _, doc in ranked[:top_k]]
    logger.info(
        "Re-ranked %d candidates → kept top %d (best=%.4f)",
        len(docs), len(top), ranked[0][0] if ranked else 0,
    )
    return top


# ─────────────────────────────────────────────────────────────────────────────
# Feature 5 — Async indexing background task
# ─────────────────────────────────────────────────────────────────────────────

async def index_file_task(
    file_path: str,
    user_id: int,
    file_id: str,
    file_name: str,
) -> None:
    """
    Full pipeline: load → split → store in Milvus.
    Writes PROCESSING → INDEXED / FAILED back to Postgres.
    """
    from src.database.config import AsyncSessionLocal
    from src.models.files import FileInputModel, IndexingStatus

    async with AsyncSessionLocal() as db:
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

            await db.execute(
                update(FileInputModel)
                .where(FileInputModel.file_id == file_id)
                .values(indexing_status=IndexingStatus.INDEXED, indexing_error=None)
            )
            await db.commit()
            logger.info("Indexing complete for file_id=%s", file_id)

        except Exception as exc:
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
# Retrieval helpers (Features 1, 2, 4)
# ─────────────────────────────────────────────────────────────────────────────

def _get_docs_with_scores(
    query: str,
    user_id: int,
    top_k: int = 4,
    fetch_k: int = 12,
) -> List[Tuple[Document, float]]:
    """
    Fetch fetch_k candidates from Milvus, re-rank with CrossEncoder,
    return top_k with their re-ranker scores.
    """
    vs         = _get_vector_store(user_id)
    candidates = vs.similarity_search(query, k=fetch_k)
    if not candidates:
        return []

    pairs  = [(query, doc.page_content) for doc in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [(doc, float(score)) for score, doc in ranked[:top_k]]


def _build_context_and_sources(
    docs_with_scores: List[Tuple[Document, float]],
) -> Tuple[str, List[Dict]]:
    """Render a context string and a deduplicated sources list."""
    context_parts = []
    sources       = []
    seen          = set()

    for doc, score in docs_with_scores:
        meta      = doc.metadata
        file_name = meta.get("file_name", "unknown")
        file_id   = meta.get("file_id", "")
        chunk_idx = meta.get("chunk_idx", 0)

        context_parts.append(
            f"[Source: {file_name}, chunk {chunk_idx}, score {score:.3f}]\n"
            f"{doc.page_content}"
        )
        key = f"{file_id}:{chunk_idx}"
        if key not in seen:
            seen.add(key)
            sources.append({"file_name": file_name, "file_id": file_id, "chunk_idx": chunk_idx})

    return "\n\n---\n\n".join(context_parts), sources


def _build_prompt(
    query: str,
    context: str,
    chat_history: List[Dict[str, str]] | None,
) -> str:
    history_block = ""
    if chat_history:
        turns = [
            f"{t.get('role','user').capitalize()}: {t.get('content','')}"
            for t in chat_history[-6:]
        ]
        history_block = "\n\n<Conversation History>\n" + "\n".join(turns) + "\n</Conversation History>"

    return (
        "You are a helpful assistant that answers questions strictly using the provided context.\n"
        "If the answer is not found in the context, say so clearly — do not invent information."
        f"{history_block}\n\n"
        f"<Context>\n{context}\n</Context>\n\n"
        f"Question: {query}\n\nAnswer:"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public API — non-streaming (Feature 3 + all above)
# ─────────────────────────────────────────────────────────────────────────────

def generate_answer(
    query: str,
    user_id: int,
    chat_history: List[Dict[str, str]] | None = None,
) -> Dict:
    """
    Full RAG pipeline (non-streaming).

    Returns:
        {"answer": str, "sources": [{"file_name", "file_id", "chunk_idx"}, ...]}
    """
    docs_with_scores = _get_docs_with_scores(query, user_id)

    if not docs_with_scores:
        return {
            "answer": "I could not find relevant information in your documents to answer this question.",
            "sources": [],
        }

    context, sources = _build_context_and_sources(docs_with_scores)
    prompt           = _build_prompt(query, context, chat_history)
    response         = llm.invoke(prompt)

    return {"answer": response.content, "sources": sources}


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 — Streaming answer
# ─────────────────────────────────────────────────────────────────────────────

async def stream_answer(
    query: str,
    user_id: int,
    chat_history: List[Dict[str, str]] | None = None,
) -> AsyncIterator[str]:
    """
    Streaming RAG pipeline.

    Yields Server-Sent Event lines:
      - One  "data: <token>\\n\\n"  per LLM token as it arrives.
      - A final "data: [SOURCES] <json>\\n\\n" so the client knows which
        documents were cited.
      - A terminal "data: [DONE]\\n\\n" to signal end-of-stream.

    Example client consumption (JavaScript):
        const es = new EventSource('/rag/stream?...');
        es.onmessage = (e) => {
            if (e.data === '[DONE]') { es.close(); return; }
            if (e.data.startsWith('[SOURCES]')) { /* parse citations */ return; }
            appendToken(e.data);
        };
    """
    import json

    # ── Retrieval (sync, fast) ────────────────────────────────────────────────
    docs_with_scores = _get_docs_with_scores(query, user_id)

    if not docs_with_scores:
        yield "data: I could not find relevant information in your documents.\n\n"
        yield "data: [DONE]\n\n"
        return

    context, sources = _build_context_and_sources(docs_with_scores)
    prompt           = _build_prompt(query, context, chat_history)

    # ── Stream tokens from LLM ────────────────────────────────────────────────
    async for chunk in llm.astream(prompt):
        token = chunk.content
        if token:
            # Escape newlines inside the SSE data field
            yield f"data: {token.replace(chr(10), ' ')}\n\n"

    # ── Send sources after all tokens ─────────────────────────────────────────
    yield f"data: [SOURCES] {json.dumps(sources)}\n\n"
    yield "data: [DONE]\n\n"