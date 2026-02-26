from pydantic import BaseModel
from typing import Optional


class UserRegisterRequest(BaseModel):
    """Request schema for user registration"""
    username: str
    password: str


class UserLoginRequest(BaseModel):
    """Request schema for user login"""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Response schema for token generation"""
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenData(BaseModel):
    """Schema for JWT token data"""
    sub: int  # user_id
    username: Optional[str] = None


class UserResponse(BaseModel):
    """Response schema for user information"""
    id: int
    username: str

    class Config:
        from_attributes = True
