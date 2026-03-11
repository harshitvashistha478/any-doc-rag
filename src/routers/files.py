"""
Files router

Medium-priority additions
──────────────────────────
  Task 8   DELETE /files/delete/{file_id} now also removes vectors from Milvus.

  Task 9   Both upload endpoints compute a SHA-256 hash of the raw bytes before
           saving.  If the same user has already uploaded an identical file, the
           request is rejected with 409 Conflict and the existing file_id is
           returned so the client can reference it immediately.
"""

import hashlib
import os
import uuid
from typing import Annotated, List

from dotenv import load_dotenv
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.config import get_db
from src.models.files import FileInputModel, IndexingStatus
from src.schemas.files import (
    FileDeleteResponse,
    FileListResponse,
    FileStatusResponse,
    FileUploadResponse,
    MultipleFileUploadResponse,
)
from src.utils.auth_dependencies import get_current_user_id
from src.utils.rag import delete_file_vectors, index_file_task

load_dotenv()

file_router = APIRouter(prefix="/files", tags=["Files"])

_raw_ext           = os.getenv("ALLOWED_EXTENSIONS", ".pdf,.txt,.docx,.md")
ALLOWED_EXTENSIONS = [e.strip() for e in _raw_ext.split(",")]
MAX_FILE_SIZE      = int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024)))
UPLOAD_DIRECTORY   = os.getenv("UPLOAD_DIRECTORY", "uploads")


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _disk_path(file_id: str, file_ext: str) -> str:
    return os.path.join(UPLOAD_DIRECTORY, f"{file_id}{file_ext}")


def _sha256(content: bytes) -> str:
    """Return the SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(content).hexdigest()


async def _check_duplicate(
    file_hash: str,
    user_id: int,
    db: AsyncSession,
) -> FileInputModel | None:
    """
    Task 9: Return the existing file record if this user has already uploaded
    an identical file (same hash), otherwise return None.
    """
    result = await db.execute(
        select(FileInputModel).where(
            (FileInputModel.file_hash == file_hash) &
            (FileInputModel.user_id == user_id)
        )
    )
    return result.scalar_one_or_none()


async def _get_owned_file(
    file_id: str,
    user_id: int,
    db: AsyncSession,
) -> FileInputModel:
    result = await db.execute(
        select(FileInputModel).where(
            (FileInputModel.file_id == file_id) &
            (FileInputModel.user_id == user_id)
        )
    )
    file = result.scalar_one_or_none()
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="File not found or you don't have permission to access it",
        )
    return file


def _validate_extension(filename: str) -> str:
    """Return the lowercased extension or raise 400."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{ext}' is not allowed. Allowed types: {', '.join(ALLOWED_EXTENSIONS)}",
        )
    return ext


def _validate_size(content: bytes) -> None:
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds the {MAX_FILE_SIZE // (1024 * 1024)} MB size limit",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Upload — single
# ─────────────────────────────────────────────────────────────────────────────

@file_router.post(
    "/upload-single",
    response_model=FileUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_single_file(
    background_tasks: BackgroundTasks,
    file: Annotated[UploadFile, File(description="One file to upload")],
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    Upload a single file.

    • Validates extension and size.
    • Task 9: Rejects with 409 if the same user already uploaded this exact file.
    • Saves to disk, records in Postgres with PENDING status.
    • Queues background RAG indexing (Feature 5).
    """
    file_ext = _validate_extension(file.filename)
    content  = await file.read()
    _validate_size(content)

    # ── Task 9: duplicate check ───────────────────────────────────────────────
    file_hash  = _sha256(content)
    duplicate  = await _check_duplicate(file_hash, user_id, db)
    if duplicate:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "You have already uploaded this exact file.",
                "existing_file_id":   duplicate.file_id,
                "existing_file_name": duplicate.file_name,
                "indexing_status":    duplicate.indexing_status,
            },
        )

    # ── Save to disk ──────────────────────────────────────────────────────────
    file_id   = str(uuid.uuid4())
    file_path = _disk_path(file_id, file_ext)
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(content)

    # ── Record in DB ──────────────────────────────────────────────────────────
    db_file = FileInputModel(
        file_name=file.filename,
        file_id=file_id,
        user_id=user_id,
        file_hash=file_hash,
        indexing_status=IndexingStatus.PENDING,
    )
    db.add(db_file)

    try:
        await db.commit()
        await db.refresh(db_file)
    except Exception as e:
        await db.rollback()
        try:
            os.remove(file_path)
        except OSError:
            pass
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    background_tasks.add_task(index_file_task, file_path, user_id, file_id, file.filename)

    return FileUploadResponse(
        success=True,
        file_id=file_id,
        file_name=file.filename,
        message="Uploaded successfully. Poll /files/status/{file_id} to track indexing.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Upload — multiple
# ─────────────────────────────────────────────────────────────────────────────

@file_router.post(
    "/upload-multiple",
    response_model=MultipleFileUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_multiple_files(
    background_tasks: BackgroundTasks,
    files: Annotated[List[UploadFile], File(description="One or more files to upload")],
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    Upload multiple files.

    Each file is independently validated.
    Task 9: Duplicate files are reported in the `failed` list (not silently skipped).
    """
    uploaded_files: list[dict] = []
    failed_files:   list[dict] = []
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

    for file in files:
        try:
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                failed_files.append({"file_name": file.filename, "error": f"'{file_ext}' not allowed"})
                continue

            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                failed_files.append({"file_name": file.filename, "error": "Exceeds size limit"})
                continue

            # ── Task 9: per-file duplicate check ─────────────────────────────
            file_hash = _sha256(content)
            duplicate = await _check_duplicate(file_hash, user_id, db)
            if duplicate:
                failed_files.append({
                    "file_name": file.filename,
                    "error":     "Duplicate — already uploaded",
                    "existing_file_id": duplicate.file_id,
                })
                continue

            file_id   = str(uuid.uuid4())
            file_path = _disk_path(file_id, file_ext)

            with open(file_path, "wb") as f:
                f.write(content)

            db.add(FileInputModel(
                file_name=file.filename,
                file_id=file_id,
                user_id=user_id,
                file_hash=file_hash,
                indexing_status=IndexingStatus.PENDING,
            ))
            uploaded_files.append({
                "file_id":   file_id,
                "file_name": file.filename,
                "file_path": file_path,
            })

        except Exception as e:
            failed_files.append({"file_name": file.filename, "error": str(e)})

    if uploaded_files:
        try:
            await db.commit()
        except Exception as e:
            await db.rollback()
            for uf in uploaded_files:
                try:
                    os.remove(uf["file_path"])
                except OSError:
                    pass
            raise HTTPException(status_code=500, detail=f"DB commit failed: {e}")

        for uf in uploaded_files:
            background_tasks.add_task(
                index_file_task, uf["file_path"], user_id, uf["file_id"], uf["file_name"]
            )

    return MultipleFileUploadResponse(
        success=len(failed_files) == 0,
        uploaded=[{"file_id": f["file_id"], "file_name": f["file_name"]} for f in uploaded_files],
        failed=failed_files,
        total_uploaded=len(uploaded_files),
        total_failed=len(failed_files),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Indexing status  (Feature 5)
# ─────────────────────────────────────────────────────────────────────────────

@file_router.get("/status/{file_id}", response_model=FileStatusResponse)
async def get_indexing_status(
    file_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Poll this after upload to track indexing progress."""
    file = await _get_owned_file(file_id, user_id, db)
    return FileStatusResponse(
        file_id=file.file_id,
        file_name=file.file_name,
        indexing_status=file.indexing_status,
        indexing_error=file.indexing_error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# List
# ─────────────────────────────────────────────────────────────────────────────

@file_router.get("/list", response_model=FileListResponse)
async def list_user_files(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    result = await db.execute(
        select(FileInputModel).where(FileInputModel.user_id == user_id)
    )
    files = result.scalars().all()
    return FileListResponse(success=True, files=files, total=len(files))


# ─────────────────────────────────────────────────────────────────────────────
# Delete  (Task 8: now also removes vectors from Milvus)
# ─────────────────────────────────────────────────────────────────────────────

@file_router.delete("/delete/{file_id}", response_model=FileDeleteResponse)
async def delete_file(
    file_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    Delete a file completely:
      1. Removes the file from disk.
      2. Task 8: Deletes all vector chunks from the user's Milvus collection.
      3. Removes the record from Postgres.

    Steps 1 and 2 are attempted even if the other fails — we always clean up
    as much as possible before removing the DB record.
    """
    file = await _get_owned_file(file_id, user_id, db)

    # ── 1. Remove from disk ───────────────────────────────────────────────────
    for ext in ALLOWED_EXTENSIONS:
        candidate = _disk_path(file_id, ext)
        if os.path.exists(candidate):
            os.remove(candidate)
            logger.info("Removed file from disk: %s", candidate)
            break

    # ── 2. Task 8: Remove vectors from Milvus ────────────────────────────────
    deleted_vectors = delete_file_vectors(user_id=user_id, file_id=file_id)
    logger.info("Removed %d vector(s) from Milvus for file_id=%s", deleted_vectors, file_id)

    # ── 3. Remove from Postgres ───────────────────────────────────────────────
    try:
        await db.delete(file)
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error during delete: {e}")

    return FileDeleteResponse(
        success=True,
        message=f"File '{file.file_name}' and its {deleted_vectors} vector(s) deleted successfully",
    )


import logging
logger = logging.getLogger(__name__)