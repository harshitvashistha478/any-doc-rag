from pydantic import BaseModel
from typing import Optional, List


class FileUploadResponse(BaseModel):
    """Response schema for single file upload"""
    success: bool
    file_id: str
    file_name: str
    message: str


class FileResponse(BaseModel):
    """Schema for file information"""
    id: int
    file_id: str
    file_name: str

    class Config:
        from_attributes = True


class FileListResponse(BaseModel):
    """Response schema for listing files"""
    success: bool
    files: List[FileResponse]
    total: int


class MultipleFileUploadResponse(BaseModel):
    """Response schema for multiple file uploads"""
    success: bool
    uploaded: List[dict]
    failed: List[dict]
    total_uploaded: int
    total_failed: int


class FileDeleteResponse(BaseModel):
    """Response schema for file deletion"""
    success: bool
    message: str
