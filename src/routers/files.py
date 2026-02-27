from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
import uuid
import os
from typing import Annotated, List

from src.database.config import get_db
from src.models.files import FileInputModel
from src.schemas.files import (
    FileUploadResponse,
    FileListResponse,
    MultipleFileUploadResponse,
    FileDeleteResponse,
)
from src.utils.auth_dependencies import get_current_user_id
from dotenv import load_dotenv
import os

load_dotenv()

file_router = APIRouter(prefix='/files', tags=['Files processing'])

ALLOWED_EXTENSIONS = os.getenv("ALLOWED_EXTENSIONS")
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE"))
UPLOAD_DIRECTORY = os.getenv("UPLOAD_DIRECTORY")


@file_router.post('/upload-single', response_model=FileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_single_file(
    file: Annotated[UploadFile, File(description="Select one file to upload")],
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    try:
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type is not valid, try uploading different extension file"
            )
        
        file_content = await file.read()
        if len(file_content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File size is to big, try uploading file size of 50 MB max"
            )
        
        file_id = str(uuid.uuid4())
        os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
        file_path = os.path.join(UPLOAD_DIRECTORY, f"{file_id}-{file_ext}")
        
        with open(file_path, 'w') as f:
            f.write()

        db_file = FileInputModel(
            file_name=file.filename,
            file_id=file_id,
            user_id=user_id
        )

        db.add(db_file)
        await db.commit()
        await db.refresh(db_file)

        return FileUploadResponse(
            success=True,
            file_id=file_id,
            file_name=file.filename,
            message="File uploaded successfully"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Something went wrong while uploading file, {str(e)}"
        )


@file_router.post('/upload-multiple', response_model=MultipleFileUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_multiple_files(
    # ✅ KEY FIX: Annotated[List[UploadFile], File(...)] tells Swagger to render file pickers
    files: Annotated[List[UploadFile], File(description="Select one or more files to upload")],
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    """
    Upload multiple files to storage and save metadata to the database.
    """
    uploaded_files = []
    failed_files = []

    os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

    for file in files:
        try:
            file_ext = os.path.splitext(file.filename)[1].lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                failed_files.append({"file_name": file.filename, "error": "File type not allowed"})
                continue

            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                failed_files.append({
                    "file_name": file.filename,
                    "error": f"File size exceeds {MAX_FILE_SIZE / (1024 * 1024):.0f}MB"
                })
                continue

            file_id = str(uuid.uuid4())
            file_path = os.path.join(UPLOAD_DIRECTORY, f"{file_id}{file_ext}")

            with open(file_path, "wb") as f:
                f.write(content)

            db_file = FileInputModel(
                file_name=file.filename,
                file_id=file_id,
                user_id=user_id
            )
            db.add(db_file)

            uploaded_files.append({
                "file_id": file_id,
                "file_name": file.filename,
                "file_path": file_path
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
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error committing files to database: {str(e)}"
        )

    return MultipleFileUploadResponse(
        success=len(failed_files) == 0,
        uploaded=uploaded_files,
        failed=failed_files,
        total_uploaded=len(uploaded_files),
        total_failed=len(failed_files)
    )


@file_router.get('/list', response_model=FileListResponse)
async def list_user_files(
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    try:
        result = await db.execute(
            select(FileInputModel).where(FileInputModel.user_id == user_id)
        )
        files = result.scalars().all()
        return FileListResponse(success=True, files=files, total=len(files))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving files: {str(e)}"
        )


@file_router.delete('/delete/{file_id}', response_model=FileDeleteResponse)
async def delete_file(
    file_id: str,
    db: AsyncSession = Depends(get_db),
    user_id: int = Depends(get_current_user_id)
):
    try:
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
                detail="File not found or you don't have permission to delete it"
            )

        await db.delete(file)
        await db.commit()

        return FileDeleteResponse(
            success=True,
            message=f"File {file_id} deleted successfully"
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting file: {str(e)}"
        )