from fastapi import APIRouter, BackgroundTasks, Depends, UploadFile, File, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import os
from typing import Annotated, List

from src.database.config import get_db
from src.models.files import FileInputModel, IndexingStatus
from src.schemas.files import (
    FileUploadResponse,
    FileListResponse,
    MultipleFileUploadResponse,
    FileDeleteResponse,
    FileStatusResponse,
)
from src.utils.auth_dependencies import get_current_user_id
from src.utils.rag import index_file_task
from dotenv import load_dotenv

load_dotenv()

file_router = APIRouter(prefix="/files", tags=["Files"])

_raw_extensions   = os.getenv("ALLOWED_EXTENSIONS", ".pdf,.txt,.docx,.md")
ALLOWED_EXTENSIONS = [ext.strip() for ext in _raw_extensions.split(",")]
MAX_FILE_SIZE      = int(os.getenv("MAX_FILE_SIZE", str(50 * 1024 * 1024)))
UPLOAD_DIRECTORY   = os.getenv("UPLOAD_DIRECTORY", "uploads")


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _disk_path(file_id: str, file_ext: str) -> str:
    return os.path.join(UPLOAD_DIRECTORY, f"{file_id}{file_ext}")


async def _get_owned_file(file_id: str, user_id: int, db: AsyncSession) -> FileInputModel:
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
    """Upload one file → save to disk → record in DB (PENDING) → queue RAG indexing."""
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"'{file_ext}' not allowed. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File exceeds {MAX_FILE_SIZE // (1024 * 1024)} MB limit",
        )

    file_id   = str(uuid.uuid4())
    file_path = _disk_path(file_id, file_ext)
    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(content)

    # Feature 5: start with PENDING status
    db_file = FileInputModel(
        file_name=file.filename,
        file_id=file_id,
        user_id=user_id,
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

    # Feature 1, 4, 5: pass user_id + file_id into the background indexing task
    background_tasks.add_task(
        index_file_task, file_path, user_id, file_id, file.filename
    )

    return FileUploadResponse(
        success=True,
        file_id=file_id,
        file_name=file.filename,
        message="Uploaded successfully. Indexing has started — poll /files/status/{file_id} for progress.",
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
    uploaded_files = []
    failed_files   = []
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

            file_id   = str(uuid.uuid4())
            file_path = _disk_path(file_id, file_ext)

            with open(file_path, "wb") as f:
                f.write(content)

            db.add(FileInputModel(
                file_name=file.filename,
                file_id=file_id,
                user_id=user_id,
                indexing_status=IndexingStatus.PENDING,
            ))
            uploaded_files.append({
                "file_id":   file_id,
                "file_name": file.filename,
                "file_path": file_path,
            })

        except Exception as e:
            failed_files.append({"file_name": file.filename, "error": str(e)})

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
# Feature 5 — Indexing status endpoint
# ─────────────────────────────────────────────────────────────────────────────

@file_router.get("/status/{file_id}", response_model=FileStatusResponse)
async def get_indexing_status(
    file_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """
    Poll this endpoint after upload to track indexing progress.

    Returns one of: pending | processing | indexed | failed
    On failure the `indexing_error` field contains the exception message.
    """
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
# Delete
# ─────────────────────────────────────────────────────────────────────────────

@file_router.delete("/delete/{file_id}", response_model=FileDeleteResponse)
async def delete_file(
    file_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id),
):
    """Delete file from disk, Postgres. (Milvus cleanup — medium priority item — tracked separately.)"""
    file = await _get_owned_file(file_id, user_id, db)

    # Remove from disk (try every allowed extension)
    for ext in ALLOWED_EXTENSIONS:
        candidate = _disk_path(file_id, ext)
        if os.path.exists(candidate):
            os.remove(candidate)
            break

    try:
        await db.delete(file)
        await db.commit()
    except Exception as e:
        await db.rollback()
        raise HTTPException(status_code=500, detail=f"DB error: {e}")

    return FileDeleteResponse(
        success=True,
        message=f"File '{file.file_name}' deleted successfully",
    )