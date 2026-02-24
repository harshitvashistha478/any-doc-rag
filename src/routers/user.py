from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from src.database.config import get_db
from src.models.users import User

user_router = APIRouter(prefix="/users", tags=["Users"])


@user_router.post("/")
async def create_user(username: str, password: str, db: AsyncSession = Depends(get_db)):
    user = User(username=username, password=password)
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@user_router.get("/")
async def get_users(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User))
    return result.scalars().all()
