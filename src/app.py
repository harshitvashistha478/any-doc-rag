from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.database import init_db
from src.routers.user import user_router
from src.routers.files import file_router
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="AnyDoc RAG Backend")

# CORS Configuration
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:8080").split(",")
CORS_ALLOW_CREDENTIALS = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router)
app.include_router(file_router)

@app.on_event("startup")
async def on_startup():
    await init_db()