"""Authentication dependencies for FastAPI routes"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import JWTError

from src.database.config import get_db
from src.models.users import User
from src.utils.jwt_utils import verify_token


security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.

    Args:
        credentials: HTTP Bearer token from request headers
        db: Database session

    Returns:
        User object from database

    Raises:
        HTTPException: If token is invalid or user not found
    """
    token = credentials.credentials

    # Verify and decode the JWT token
    try:
        payload = verify_token(token)
    except JWTError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication credentials: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract sub claim (JWT spec stores it as a string)
    sub = payload.get("sub")
    if sub is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: missing user_id",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Convert sub back to int for database lookup
    try:
        user_id: int = int(sub)
    except (ValueError, TypeError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token: user_id is malformed",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Verify user exists in database
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_user_id(
    current_user: User = Depends(get_current_user)
) -> int:
    """
    Get current user ID from authenticated user.

    Args:
        current_user: Current authenticated user

    Returns:
        User ID as integer
    """
    return current_user.id