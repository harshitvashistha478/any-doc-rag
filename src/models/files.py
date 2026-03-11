import enum
from sqlalchemy import Column, Integer, String, ForeignKey, Enum, Text
from sqlalchemy.orm import relationship
from src.database.config import Base


class IndexingStatus(str, enum.Enum):
    PENDING    = "pending"     # just uploaded, not yet indexed
    PROCESSING = "processing"  # background task is running
    INDEXED    = "indexed"     # successfully stored in Milvus
    FAILED     = "failed"      # indexing threw an exception


class FileInputModel(Base):
    __tablename__ = "files"

    id               = Column(Integer, autoincrement=True, primary_key=True, index=True)
    file_name        = Column(String(255), nullable=False)
    file_id          = Column(String, unique=True, index=True, nullable=False)
    user_id          = Column(Integer, ForeignKey("users.id"), nullable=False)

    # ── Feature 5: indexing status tracking ──────────────────────────────────
    indexing_status  = Column(
        Enum(IndexingStatus),
        nullable=False,
        default=IndexingStatus.PENDING,
        server_default=IndexingStatus.PENDING.value,
    )
    indexing_error   = Column(Text, nullable=True)   # populated only on FAILED

    user = relationship("User", back_populates="files")