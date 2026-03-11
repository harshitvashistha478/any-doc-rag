import re
from pydantic import BaseModel, Field, field_validator
from typing import Optional


class UserRegisterRequest(BaseModel):
    """Request schema for user registration — Task 10: full input validation."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="3–50 characters, letters/digits/underscores only",
    )
    password: str = Field(
        ...,
        min_length=8,
        max_length=128,
        description="8–128 characters, must include uppercase, lowercase, digit, special char",
    )

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        """Only letters, digits, and underscores allowed — no spaces or special chars."""
        if not re.match(r"^[a-zA-Z0-9_]+$", v):
            raise ValueError(
                "Username may only contain letters, digits, and underscores"
            )
        return v

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        """Enforce minimum password complexity."""
        errors = []
        if not re.search(r"[A-Z]", v):
            errors.append("at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            errors.append("at least one lowercase letter")
        if not re.search(r"\d", v):
            errors.append("at least one digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>_\-+=\[\]\\;'/`~]", v):
            errors.append("at least one special character (!@#$%^&* etc.)")
        if errors:
            raise ValueError("Password must contain: " + ", ".join(errors))
        return v


class UserLoginRequest(BaseModel):
    """Request schema for user login."""
    username: str
    password: str


class TokenResponse(BaseModel):
    """Response schema for token generation."""
    access_token: str
    token_type:   str = "bearer"
    expires_in:   int


class TokenData(BaseModel):
    """Schema for JWT token data."""
    sub:      int             # user_id
    username: Optional[str] = None


class UserResponse(BaseModel):
    """Response schema for user information."""
    id:       int
    username: str

    class Config:
        from_attributes = True