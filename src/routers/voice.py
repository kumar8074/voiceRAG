# ===================================================================================
# Project: VoiceRAG
# File: src/routers/voice.py
# Description: FastAPI WebSocket endpoint for real-time voice pipeline.
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from __future__ import annotations

import asyncio
import json
import os

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..services.voice.asr_handler import ASRSession, ASREvent, ASREventType
from ..services.voice.pipeline    import run_pipeline
from ..logger import logging

router = APIRouter(tags=["voice_ws"])

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")


# WebSocket frame helper 
async def _send(ws: WebSocket, **kwargs) -> None:
    try:
        await ws.send_text(json.dumps(kwargs))
    except Exception:
        pass



#  WebSocket endpoint
@router.websocket("/ws/voice")
async def voice_websocket(
    ws:         WebSocket,
    user_id:    str,
    session_id: str,
):
    """
    Persistent WebSocket for the full real-time voice pipeline.

    Browser → Server:
        Binary frames : raw PCM 16kHz s16le chunks (from AudioWorklet)
        Text frames   : { "type": "stop" }

    Server → Browser:
        { "type": "ready" }
        { "type": "listening" }
        { "type": "speech_end" }
        { "type": "transcript", "text": "…", "language_code": "hi-IN"|"en-IN" }
        { "type": "rag_start" }
        { "type": "token",     "text": "…" }
        { "type": "rag_done" }
        { "type": "tts_start", "language_code": "…" }
        { "type": "tts_done" }
        { "type": "barge_in" }
        { "type": "error",     "message": "…" }
        Binary frames : raw MP3 chunks (TTS output)
    """
    await ws.accept()
    logging.info(f"[WS] Connected: user={user_id} session={session_id}")

    if not SARVAM_API_KEY:
        await _send(ws, type="error", message="SARVAM_API_KEY not configured.")
        await ws.close()
        return

    await _send(ws, type="ready")

    # Shared state 
    audio_queue:   asyncio.Queue[bytes | None] = asyncio.Queue(maxsize=512)
    cancel_event:  asyncio.Event               = asyncio.Event()
    pipeline_task: asyncio.Task | None         = None

    # ASR event callback 
    # Called by ASRSession for every speech_start / speech_end / transcript.
    # This is the only place barge-in logic lives in the router.
    async def on_asr_event(event: ASREvent) -> None:
        nonlocal pipeline_task, cancel_event

        if event.type == ASREventType.SPEECH_START:
            # Barge-in: cancel running pipeline immediately 
            if pipeline_task and not pipeline_task.done():
                logging.info("[WS] Barge-in — cancelling pipeline.")
                cancel_event.set()
                pipeline_task.cancel()
                try:
                    await pipeline_task
                except (asyncio.CancelledError, Exception):
                    pass
                cancel_event  = asyncio.Event()   # fresh for next turn
                pipeline_task = None
                await _send(ws, type="barge_in")

            await _send(ws, type="listening")

        elif event.type == ASREventType.SPEECH_END:
            await _send(ws, type="speech_end")

        elif event.type == ASREventType.TRANSCRIPT:
            if not event.transcript:
                await _send(ws, type="listening")
                return

            # Forward transcript to browser for display
            await _send(
                ws,
                type="transcript",
                text=event.transcript,
                language_code=event.language_code,
            )

            # Launch RAG → TTS pipeline as a background task
            cancel_event  = asyncio.Event()
            pipeline_task = asyncio.create_task(
                run_pipeline(
                    ws=ws,
                    transcript=event.transcript,
                    language_code=event.language_code,
                    user_id=user_id,
                    session_id=session_id,
                    cancel_event=cancel_event,
                )
            )

    # Browser receive loop 
    async def receive_browser_loop() -> None:
        chunks_received = 0
        try:
            while True:
                message = await ws.receive()

                if message.get("bytes"):
                    chunks_received += 1
                    # Log every 50 chunks to confirm browser→server flow
                    if chunks_received % 50 == 0:
                        logging.debug(
                            f"[WS] Browser→server: {chunks_received} PCM chunks received"
                        )
                    await audio_queue.put(message["bytes"])

                elif message.get("text"):
                    data = json.loads(message["text"])
                    if data.get("type") == "stop":
                        logging.info(
                            f"[WS] Stop signal. Total PCM chunks received: {chunks_received}"
                        )
                        await audio_queue.put(None)
                        break

        except WebSocketDisconnect:
            logging.info("[WS] Client disconnected.")
            await audio_queue.put(None)
        except Exception as exc:
            logging.error(f"[WS] Receive error: {exc}")
            await audio_queue.put(None)

    # Run ASR session + browser loop concurrently 
    asr_task     = asyncio.create_task(
        ASRSession().run(audio_queue, on_asr_event)
    )
    browser_task = asyncio.create_task(receive_browser_loop())

    try:
        await asyncio.gather(asr_task, browser_task)
    except Exception as exc:
        logging.error(f"[WS] Top-level error: {exc}")
    finally:
        if pipeline_task and not pipeline_task.done():
            cancel_event.set()
            pipeline_task.cancel()
            try:
                await pipeline_task
            except (asyncio.CancelledError, Exception):
                pass

        asr_task.cancel()
        browser_task.cancel()
        logging.info(f"[WS] Session closed: user={user_id} session={session_id}")