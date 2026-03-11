"""
RAG router

Medium-priority additions
──────────────────────────
  Task 6   POST /rag/stream                    — stateless streaming query
           POST /rag/sessions/{id}/stream      — session-aware streaming query

Both streaming endpoints use FastAPI's StreamingResponse with Server-Sent Events
(SSE).  The LLM tokens stream in real-time; sources arrive in a final event
before the [DONE] sentinel.

SSE event format
────────────────
  data: <token text>\\n\\n          — one per LLM token
  data: [SOURCES] <json array>\\n\\n — citations, sent after all tokens
  data: [DONE]\\n\\n                 — signals end of stream
"""

import json

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from typing import AsyncIterator, List

from src.database.config import get_db
from src.models.chat import ChatMessage, ChatSession, MessageRole
from src.utils.auth_dependencies import get_current_user, get_current_user_id
from src.utils.rag import generate_answer, stream_answer

rag_router = APIRouter(prefix="/rag", tags=["RAG"])


# ─────────────────────────────────────────────────────────────────────────────
# Schemas
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
# Internal helpers
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


async def _load_history(session_id: int, db: AsyncSession) -> list[dict]:
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.id)
    )
    return [{"role": m.role.value, "content": m.content} for m in result.scalars().all()]


# ─────────────────────────────────────────────────────────────────────────────
# Session management  (Feature 3)
# ─────────────────────────────────────────────────────────────────────────────

@rag_router.post("/sessions", response_model=SessionSchema, status_code=status.HTTP_201_CREATED)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
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
    session = await _get_owned_session(session_id, user_id, db)
    result  = await db.execute(
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
    session = await _get_owned_session(session_id, user_id, db)
    await db.delete(session)
    await db.commit()


# ─────────────────────────────────────────────────────────────────────────────
# Non-streaming queries
# ─────────────────────────────────────────────────────────────────────────────

@rag_router.post("/sessions/{session_id}/query", response_model=RAGQueryResponse)
async def session_query(
    session_id: int,
    body: SessionQueryRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Multi-turn query — loads history, answers, persists both turns."""
    await _get_owned_session(session_id, user_id, db)
    history = await _load_history(session_id, db)

    try:
        result_data = generate_answer(query=body.query, user_id=user_id, chat_history=history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    db.add(ChatMessage(session_id=session_id, role=MessageRole.USER,      content=body.query))
    db.add(ChatMessage(session_id=session_id, role=MessageRole.ASSISTANT, content=result_data["answer"]))
    await db.commit()

    return RAGQueryResponse(
        success=True,
        query=body.query,
        answer=result_data["answer"],
        sources=[SourceSchema(**s) for s in result_data["sources"]],
    )


@rag_router.post("/query", response_model=RAGQueryResponse)
async def stateless_query(
    body: RAGQueryRequest,
    user_id: int = Depends(get_current_user_id),
):
    """One-off query with no session history."""
    try:
        result_data = generate_answer(query=body.query, user_id=user_id, chat_history=None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG error: {e}")

    return RAGQueryResponse(
        success=True,
        query=body.query,
        answer=result_data["answer"],
        sources=[SourceSchema(**s) for s in result_data["sources"]],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Task 6 — Streaming endpoints
# ─────────────────────────────────────────────────────────────────────────────

@rag_router.post("/stream")
async def stateless_stream(
    body: RAGQueryRequest,
    user_id: int = Depends(get_current_user_id),
):
    """
    Stateless streaming query.  Returns tokens as Server-Sent Events.

    SSE protocol:
      data: <token>\\n\\n
      data: [SOURCES] [{"file_name":..., "file_id":..., "chunk_idx":...}, ...]\\n\\n
      data: [DONE]\\n\\n
    """
    async def event_generator() -> AsyncIterator[str]:
        async for chunk in stream_answer(
            query=body.query,
            user_id=user_id,
            chat_history=None,
        ):
            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            # Disable buffering so tokens arrive at the client in real-time
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@rag_router.post("/sessions/{session_id}/stream")
async def session_stream(
    session_id: int,
    body: SessionQueryRequest,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    Session-aware streaming query.

    - Loads the full conversation history before streaming.
    - After the stream is exhausted, persists both turns to the DB so the
      next query in this session sees them.

    SSE protocol: same as /rag/stream
    """
    await _get_owned_session(session_id, user_id, db)
    history = await _load_history(session_id, db)

    # Collect the full answer while streaming so we can persist it afterward
    answer_parts: list[str] = []
    sources_data: list[dict] = []

    async def event_generator() -> AsyncIterator[str]:
        async for chunk in stream_answer(
            query=body.query,
            user_id=user_id,
            chat_history=history,
        ):
            # Intercept the [SOURCES] event to capture citation data
            if chunk.startswith("data: [SOURCES]"):
                try:
                    payload = chunk.removeprefix("data: [SOURCES] ").strip()
                    sources_data.extend(json.loads(payload))
                except Exception:
                    pass
            elif chunk.startswith("data: [DONE]"):
                # Persist both turns once streaming is complete
                full_answer = "".join(answer_parts)
                db.add(ChatMessage(
                    session_id=session_id,
                    role=MessageRole.USER,
                    content=body.query,
                ))
                db.add(ChatMessage(
                    session_id=session_id,
                    role=MessageRole.ASSISTANT,
                    content=full_answer,
                ))
                try:
                    await db.commit()
                except Exception:
                    await db.rollback()
            elif not chunk.startswith("data: ["):
                # Regular token — strip the "data: " prefix and collect
                token = chunk.removeprefix("data: ").rstrip("\n")
                answer_parts.append(token)

            yield chunk

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
        },
    )