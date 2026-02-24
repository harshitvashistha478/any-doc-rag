from src.database.config import engine, Base
from src.models import users  

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)