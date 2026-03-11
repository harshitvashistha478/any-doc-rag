import enum
from sqlalchemy import Column, Integer, String, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship
from src.database.config import Base


class IndexingStatus(str, enum.Enum):
    PENDING    = "pending"
    PROCESSING = "processing"
    INDEXED    = "indexed"
    FAILED     = "failed"


class FileInputModel(Base):
    __tablename__ = "files"

    id              = Column(Integer, autoincrement=True, primary_key=True, index=True)
    file_name       = Column(String(255), nullable=False)
    file_id         = Column(String, unique=True, index=True, nullable=False)
    user_id         = Column(Integer, ForeignKey("users.id"), nullable=False)
    indexing_status = Column(
        Enum(IndexingStatus),
        nullable=False,
        default=IndexingStatus.PENDING,
        server_default=IndexingStatus.PENDING.value,
    )
    indexing_error  = Column(Text, nullable=True)

    # ── Task 9: duplicate detection ──────────────────────────────────────────
    # SHA-256 hex digest of raw file bytes. Indexed for fast lookups.
    # Scoped per-user — two users CAN independently upload the same file.
    file_hash       = Column(String(64), nullable=True, index=True)

    user = relationship("User", back_populates="files")