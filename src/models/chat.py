from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, ForeignKey, Text, DateTime, Enum
from sqlalchemy.orm import relationship
import enum

from src.database.config import Base


class MessageRole(str, enum.Enum):
    USER      = "user"
    ASSISTANT = "assistant"


class ChatSession(Base):
    """
    A named conversation thread belonging to one user.
    Each session maintains its own message history for multi-turn RAG.
    """
    __tablename__ = "chat_sessions"

    id         = Column(Integer, autoincrement=True, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False)
    title      = Column(String(255), nullable=False, default="New Chat")
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True),
                        default=lambda: datetime.now(timezone.utc),
                        onupdate=lambda: datetime.now(timezone.utc))

    user     = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session",
                            cascade="all, delete-orphan", order_by="ChatMessage.id")


class ChatMessage(Base):
    """
    A single turn (user or assistant) inside a ChatSession.
    """
    __tablename__ = "chat_messages"

    id         = Column(Integer, autoincrement=True, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False)
    role       = Column(Enum(MessageRole), nullable=False)
    content    = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    session = relationship("ChatSession", back_populates="messages")