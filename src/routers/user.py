from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from datetime import timedelta

from src.database.config import get_db
from src.models.users import User
from src.schemas.auth import UserRegisterRequest, UserLoginRequest, TokenResponse, UserResponse
from src.utils.jwt_utils import (
    hash_password,
    verify_password,
    create_access_token,
    ACCESS_TOKEN_EXPIRE_MINUTES
)
from src.utils.auth_dependencies import get_current_user

user_router = APIRouter(prefix="/users", tags=["Users"])


@user_router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register_user(
    register_data: UserRegisterRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user.

    Args:
        register_data: Username and password in request body
        db: Database session

    Returns:
        UserResponse with created user details

    Raises:
        HTTPException: If username already exists
    """
    # Check if user already exists
    result = await db.execute(select(User).where(User.username == register_data.username))
    existing_user = result.scalar_one_or_none()

    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )

    # Hash password and create user
    hashed_password = hash_password(register_data.password)
    user = User(username=register_data.username, password=hashed_password)

    db.add(user)
    await db.commit()
    await db.refresh(user)

    return user


@user_router.post("/login", response_model=TokenResponse)
async def login(
    login_data: UserLoginRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Authenticate user and return JWT token.

    Args:
        login_data: Username and password
        db: Database session

    Returns:
        TokenResponse with JWT access token

    Raises:
        HTTPException: If credentials are invalid
    """
    # Find user by username
    result = await db.execute(select(User).where(User.username == login_data.username))
    user = result.scalar_one_or_none()

    if not user or not verify_password(login_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Create access token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.id, "username": user.username},
        expires_delta=access_token_expires
    )

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        expires_in=ACCESS_TOKEN_EXPIRE_MINUTES * 60  # Convert to seconds
    )


@user_router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user information.

    Args:
        current_user: Current authenticated user from JWT token

    Returns:
        UserResponse with current user details
    """
    return current_user


@user_router.get("/", response_model=list[UserResponse])
async def get_users(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get all users (only for authenticated users).

    Args:
        db: Database session
        current_user: Current authenticated user

    Returns:
        List of users
    """
    result = await db.execute(select(User))
    return result.scalars().all()
