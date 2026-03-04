# ===================================================================================
# Project: VoiceRAG
# File: src/routers/chat.py
# Description: Chat Endpoint 
# Author: LALAN KUMAR
# Created: [04-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from ..dependencies import get_db_session
from ..services.rag.rag_service import RAGService
from ..logger import logging
from ..schemas.api.request_response import ChatRequest

router = APIRouter(prefix="/api/v1", tags=["chat"])

# Singleton — one instance shared across all requests
_rag_service = RAGService()


@router.post("/chat")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db_session)
):
    """
    Streaming chat endpoint.
    Returns a text/event-stream response with tokens as they arrive.
    """

    if not request.query or not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    if not request.user_id or not request.session_id:
        raise HTTPException(status_code=400, detail="user_id and session_id are required")

    logging.info(f"Chat request: user={request.user_id}, session={request.session_id}")

    async def token_stream():
        try:
            async for token in _rag_service.query(
                query=request.query,
                user_id=request.user_id,
                session_id=request.session_id,
                db=db,
                top_k=request.top_k
            ):
                yield token
        except Exception as e:
            logging.error(f"Streaming error: {e}")
            yield f"\n[Error: {str(e)}]"

    return StreamingResponse(
        token_stream(),
        media_type="text/plain",
        headers={
            "X-Accel-Buffering": "no",   # disable nginx buffering
            "Cache-Control": "no-cache",
        }
    )