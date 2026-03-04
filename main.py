# ===================================================================================
# Project: VoiceRAG
# File: main.py
# Description: FastAPI application entry point
# Author: LALAN KUMAR
# Created: [02-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from contextlib import asynccontextmanager
import os
import uvicorn

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.db.init_db import init_db
from src.dependencies import get_qdrant_client, get_embedding_service
from src.routers.file_upload import router as file_upload_router
from src.routers.chat import router as chat_router
from src.routers.voice import router as voice_ws_router 
from src.logger import logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: init DB tables, warm up singletons.
    """
    logging.info("Starting VoiceRAG...")

    # Init Postgres tables
    await init_db()
    logging.info("DB tables ready.")

    # Warm up singletons at startup — not on first request
    get_qdrant_client()
    logging.info("Qdrant client ready.")

    get_embedding_service()
    logging.info("Embedding service ready.")

    yield

    logging.info("VoiceRAG shutting down.")


app = FastAPI(
    title="VoiceRAG",
    description="Multilingual voice-enabled RAG system",
    version="1.0.0",
    lifespan=lifespan
)

# Routers
app.include_router(file_upload_router)
app.include_router(chat_router)
app.include_router(voice_ws_router)  

# Static files & UI
os.makedirs("tmp", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def serve_ui():
    return FileResponse("static/index.html")


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="localhost",
        port=8000,
        #reload=True
    )