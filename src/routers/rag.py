"""
RAG router — Feature 3 (conversation memory) exposed as REST endpoints.

Endpoints
─────────
POST /rag/sessions                       Create a new chat session
GET  /rag/sessions                       List all sessions for the current user
GET  /rag/sessions/{id}                  Get a session + its full message history
DELETE /rag/sessions/{id}                Delete session and all its messages

POST /rag/sessions/{id}/query            Ask a question inside a session
                                         (history is loaded automatically)

POST /rag/query                          Stateless one-off query (no session)
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional

from src.database.config import get_db
from src.models.chat import ChatSession, ChatMessage, MessageRole
from src.utils.auth_dependencies import get_current_user_id
from src.utils.rag import generate_answer

rag_router = APIRouter(prefix="/rag", tags=["RAG"])


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic schemas  (kept here since they're router-specific)
# ─────────────────────────────────────────────────────────────────────────────

class SourceSchema(BaseModel):
    file_name: str
    file_id:   str
    chunk_idx: int


class RAGQueryRequest(BaseModel):
    query:      str  = Field(..., min_length=1)
    use_scores: bool = Field(False)


class RAGQueryResponse(BaseModel):
    success: bool
    query:   str
    answer:  str
    sources: List[SourceSchema] = []


class CreateSessionRequest(BaseModel):
    title: str = Field("New Chat", min_length=1, max_length=255)


class MessageSchema(BaseModel):
    id:      int
    role:    str
    content: str

    class Config:
        from_attributes = True


class SessionSchema(BaseModel):
    id:       int
    title:    str
    messages: List[MessageSchema] = []

    class Config:
        from_attributes = True


class SessionQueryRequest(BaseModel):
    query: str = Field(..., min_length=1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

async def _get_owned_session(
    session_id: int,
    user_id: int,
    db: AsyncSession,
) -> ChatSession:
    result = await db.execute(
        select(ChatSession).where(
            (ChatSession.id == session_id) &
            (ChatSession.user_id == user_id)
        )
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or you don't have access to it",
        )
    return session


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — Session management
# ─────────────────────────────────────────────────────────────────────────────

@rag_router.post("/sessions", response_model=SessionSchema, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Create a new named chat session."""
    session = ChatSession(user_id=user_id, title=body.title)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    return session


@rag_router.get("/sessions", response_model=List[SessionSchema])
async def list_sessions(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """List all chat sessions for the current user (newest first)."""
    result = await db.execute(
        select(ChatSession)
        .where(ChatSession.user_id == user_id)
        .order_by(ChatSession.updated_at.desc())
    )
    return result.scalars().all()


@rag_router.get("/sessions/{session_id}", response_model=SessionSchema)
async def get_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Get a session and its full message history."""
    session = await _get_owned_session(session_id, user_id, db)

    # Eagerly load messages
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
    )
    session.messages = result.scalars().all()
    return session


@rag_router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_session(
    session_id: int,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Delete a session and all its messages (cascade)."""
    session = await _get_owned_session(session_id, user_id, db)
    await db.delete(session)
    await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Feature 3 — Multi-turn query inside a session
# ─────────────────────────────────────────────────────────────────────────────

@rag_router.post("/sessions/{session_id}/query", response_model=RAGQueryResponse)
async def session_query(
    session_id: int,
    body: SessionQueryRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    Ask a question inside a session.

    - The full message history of the session is loaded and passed to the LLM
      so it can resolve pronouns and handle follow-up questions.
    - Both the user turn and the assistant response are saved back to the DB
      so the next query in this session will see them.
    """
    await _get_owned_session(session_id, user_id, db)

    # Load existing messages to build history
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
    )
    messages = result.scalars().all()
    history  = [{"role": m.role.value, "content": m.content} for m in messages]

    # Generate answer with full history (Feature 3) + per-user retrieval (Feature 1&2)
    # + re-ranking (Feature 4) — all inside generate_answer()
    try:
        result_data = generate_answer(
            query=body.query,
            user_id=user_id,
            chat_history=history,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    # Persist both turns to the session
    db.add(ChatMessage(session_id=session_id, role=MessageRole.USER,      content=body.query))
    db.add(ChatMessage(session_id=session_id, role=MessageRole.ASSISTANT, content=result_data["answer"]))
    await db.commit()

    return RAGQueryResponse(
        success=True,
        query=body.query,
        answer=result_data["answer"],
        sources=[SourceSchema(**s) for s in result_data["sources"]],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Stateless one-off query (no session needed)
# ─────────────────────────────────────────────────────────────────────────────

@rag_router.post("/query", response_model=RAGQueryResponse)
async def stateless_query(
    body: RAGQueryRequest,
    user_id: int = Depends(get_current_user_id),
):
    """
    Quick one-off query — no session, no history.
    Good for single questions; use /sessions/{id}/query for conversations.
    """
    try:
        result_data = generate_answer(
            query=body.query,
            user_id=user_id,
            chat_history=None,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    return RAGQueryResponse(
        success=True,
        query=body.query,
        answer=result_data["answer"],
        sources=[SourceSchema(**s) for s in result_data["sources"]],
    )