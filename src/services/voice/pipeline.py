# ===================================================================================
# Project: VoiceRAG
# File: src/services/voice/pipeline.py
# Description: Orchestrates the RAG → TTS pipeline for a single voice turn
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from __future__ import annotations

import asyncio
import json
from fastapi import WebSocket

from ...db.factory import async_session
from ...services.rag.rag_service import RAGService
from ...logger import logging
from .tts_handler import synthesize_stream


_rag = RAGService()


# WebSocket frame helpers 
async def _send(ws: WebSocket, **kwargs) -> None:
    """Send a typed JSON control frame to the browser."""
    try:
        await ws.send_text(json.dumps(kwargs))
    except Exception:
        pass   # browser may have disconnected


async def _send_bytes(ws: WebSocket, data: bytes) -> None:
    """Send raw binary (MP3 chunk) to the browser."""
    try:
        await ws.send_bytes(data)
    except Exception:
        pass


async def run_pipeline(
    ws:            WebSocket,
    transcript:    str,
    language_code: str,
    user_id:       str,
    session_id:    str,
    cancel_event:  asyncio.Event,
) -> None:
    """
    Run one full voice turn: RAG streaming → TTS streaming.

    Args:
        ws:            FastAPI WebSocket — frames sent directly to browser.
        transcript:    Final ASR transcript for this turn.
        language_code: hi-IN or en-IN — used for TTS voice selection.
        user_id:       For RAG multi-tenant isolation + DB persistence.
        session_id:    For RAG history + Qdrant filtering.
        cancel_event:  Set externally to abort at the next await point.
    """
    await _send(ws, type="rag_start")
    logging.info(f"[Pipeline] RAG start | query='{transcript[:60]}…'")

    full_response: list[str] = []

    try:
        async with async_session() as db:
            async for token in _rag.query(
                query=transcript,
                user_id=user_id,
                session_id=session_id,
                db=db,
                top_k=5,
            ):
                if cancel_event.is_set():
                    logging.info("[Pipeline] Barge-in during RAG — aborting.")
                    await _send(ws, type="barge_in")
                    return

                full_response.append(token)
                # Stream token to browser for live display in the voice modal
                await _send(ws, type="token", text=token)

    except asyncio.CancelledError:
        logging.info("[Pipeline] RAG task cancelled (barge-in).")
        return

    except Exception as exc:
        logging.error(f"[Pipeline] RAG error: {exc}")
        await _send(ws, type="error", message=f"RAG error: {exc}")
        return

    if cancel_event.is_set():
        return

    response_text = "".join(full_response).strip()

    await _send(ws, type="rag_done")
    logging.info(
        f"[Pipeline] RAG done | "
        f"{len(full_response)} tokens, {len(response_text)} chars"
    )

    if not response_text:
        logging.warning("[Pipeline] Empty RAG response — skipping TTS.")
        return

    await _send(ws, type="tts_start", language_code=language_code)
    logging.info(f"[Pipeline] TTS start | lang={language_code}")

    chunk_count = 0
    try:
        async for mp3_chunk in synthesize_stream(
            text=response_text,
            language_code=language_code,
            cancel_event=cancel_event,
        ):
            if cancel_event.is_set():
                logging.info("[Pipeline] Barge-in during TTS — aborting.")
                await _send(ws, type="barge_in")
                return

            chunk_count += 1
            await _send_bytes(ws, mp3_chunk)

    except asyncio.CancelledError:
        logging.info("[Pipeline] TTS task cancelled (barge-in).")
        return

    except RuntimeError as exc:
        logging.error(f"[Pipeline] TTS error: {exc}")
        await _send(ws, type="error", message=f"TTS error: {exc}")
        return

    if not cancel_event.is_set():
        await _send(ws, type="tts_done")
        logging.info(
            f"[Pipeline] Turn complete | "
            f"{chunk_count} TTS chunks sent"
        )