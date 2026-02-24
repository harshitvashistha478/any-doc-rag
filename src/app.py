from fastapi import FastAPI
from src.database import init_db
from src.routers.user import user_router

app = FastAPI(title="AnyDoc RAG Backend")

app.include_router(user_router)

@app.on_event("startup")
async def on_startup():
    await init_db()