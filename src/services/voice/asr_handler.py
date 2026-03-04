# ===================================================================================
# Project: VoiceRAG
# File: src/services/voice/asr_handler.py
# Description: Sarvam Saaras v3 streaming ASR WebSocket session.
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from enum import Enum, auto
from typing import Awaitable, Callable

from sarvamai import AsyncSarvamAI

from ...logger import logging
from .audio_utils import pcm_to_base64, detect_language

# Constants 

SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY", "")
ASR_MODEL       = "saaras:v3"
ASR_SAMPLE_RATE = 16000        # MUST match what the browser worklet sends
                               # Browser resamples to exactly this rate before sending

# transcribe mode: keeps output in source language (Hindi stays Hindi, English stays English)
# codemix would mix scripts in output which makes script-based language detection harder
ASR_MODE        = "transcribe"

# hi-IN is used as the connection language. Saaras v3 with high_vad_sensitivity
# handles English speech well even when connected as hi-IN — the transcript
# output will contain English text which our script detector correctly routes
# to en-IN for TTS.
ASR_LANGUAGE    = "hi-IN"
ASR_CODEC       = "pcm_s16le"  # raw signed-16-bit little-endian PCM


# Event types 
class ASREventType(Enum):
    SPEECH_START = auto()
    SPEECH_END   = auto()
    TRANSCRIPT   = auto()


@dataclass
class ASREvent:
    type:          ASREventType
    transcript:    str = ""
    language_code: str = "en-IN"   # always hi-IN or en-IN, set by detect_language()


class ASRSession:
    """
    Manages a Sarvam Saaras v3 streaming ASR WebSocket session.

    The session runs until a None sentinel is placed in audio_queue
    or the Sarvam WebSocket closes.

    event_callback fires for every speech_start, speech_end, transcript.
    Language in transcript events is detected via Unicode script analysis
    (not from ASR response — saaras:v3 STT does not return language_code).
    """

    def __init__(self, api_key: str = SARVAM_API_KEY):
        if not api_key:
            raise ValueError("SARVAM_API_KEY is not set.")
        self._api_key = api_key

    async def run(
        self,
        audio_queue:    asyncio.Queue,
        event_callback: Callable[[ASREvent], Awaitable[None]],
    ) -> None:
        client = AsyncSarvamAI(api_subscription_key=self._api_key)

        async with client.speech_to_text_streaming.connect(
            model=ASR_MODEL,
            mode=ASR_MODE,
            language_code=ASR_LANGUAGE,
            sample_rate=ASR_SAMPLE_RATE,     # ← MUST match transcribe() call below
            input_audio_codec=ASR_CODEC,
            high_vad_sensitivity=True,       # 0.5 s silence → speech_end
            vad_signals=True,                # receive speech_start / speech_end
        ) as asr_ws:

            logging.info(
                f"[ASR] Connected — model={ASR_MODEL} mode={ASR_MODE} "
                f"lang={ASR_LANGUAGE} rate={ASR_SAMPLE_RATE}Hz"
            )

            send_task = asyncio.create_task(
                self._send_loop(asr_ws, audio_queue)
            )
            recv_task = asyncio.create_task(
                self._recv_loop(asr_ws, event_callback)
            )

            try:
                done, pending = await asyncio.wait(
                    [send_task, recv_task],
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                    try:
                        await t
                    except (asyncio.CancelledError, Exception):
                        pass
            except Exception as exc:
                logging.error(f"[ASR] Session error: {exc}")
                raise RuntimeError(f"ASR session error: {exc}") from exc
            finally:
                send_task.cancel()
                recv_task.cancel()
                logging.info("[ASR] Session closed.")

    # Send loop
    @staticmethod
    async def _send_loop(asr_ws, audio_queue: asyncio.Queue) -> None:
        """
        Pull raw PCM chunks from the queue and forward to Sarvam.

        IMPORTANT: sample_rate MUST be passed here too — Sarvam requires it
        in both the connection params and each individual transcribe() call.
        Omitting it (or passing the wrong value) causes silent VAD.
        """
        chunks_sent  = 0
        bytes_sent   = 0

        while True:
            chunk = await audio_queue.get()

            if chunk is None:
                logging.info(
                    f"[ASR] Send loop: sentinel received — "
                    f"sent {chunks_sent} chunks / {bytes_sent} bytes total. Flushing."
                )
                try:
                    await asr_ws.flush()
                except Exception:
                    pass
                break

            chunks_sent += 1
            bytes_sent  += len(chunk)

            # Log every 50 chunks (~1.6 s at 16kHz/128-frame worklet) so we can
            # confirm audio is flowing without flooding the log
            if chunks_sent % 50 == 0:
                logging.debug(
                    f"[ASR] Send loop: {chunks_sent} chunks / "
                    f"{bytes_sent/1024:.1f} KB sent to Sarvam so far"
                )

            encoded = pcm_to_base64(chunk)
            await asr_ws.transcribe(
                audio=encoded,
                encoding="audio/wav",
                sample_rate=ASR_SAMPLE_RATE,   # ← MUST match connection sample_rate
            )

    # Receive loop 
    @staticmethod
    async def _recv_loop(
        asr_ws,
        event_callback: Callable[[ASREvent], Awaitable[None]],
    ) -> None:
        """
        Process VAD and transcript events from Sarvam.

        Language is detected from the transcript text using Unicode script
        analysis — NOT from message.get("language_code") which is absent
        in saaras:v3 STT transcript responses.
        """
        messages_received = 0

        async for message in asr_ws:
            messages_received += 1
            logging.debug(f"[ASR] Raw message #{messages_received}: {message!r}")

            # Sarvam SDK returns SpeechToTextStreamingResponse objects, not dicts.
            # Structure:
            #   type='events' → data.signal_type in ('START_SPEECH', 'END_SPEECH')
            #   type='data'   → data.transcript  (the final transcription)
            msg_type    = getattr(message, "type", None)
            msg_data    = getattr(message, "data", None)
            signal_type = getattr(msg_data, "signal_type", None)

            if msg_type == "events" and signal_type == "START_SPEECH":
                logging.info("[ASR] VAD: speech_start")
                await event_callback(ASREvent(type=ASREventType.SPEECH_START))

            elif msg_type == "events" and signal_type == "END_SPEECH":
                logging.info("[ASR] VAD: speech_end")
                await event_callback(ASREvent(type=ASREventType.SPEECH_END))

            elif msg_type == "data":
                text = getattr(msg_data, "transcript", "")
                if isinstance(text, str):
                    text = text.strip()

                if not text:
                    logging.info("[ASR] Empty transcript — skipping.")
                    continue

                lang = detect_language(text)
                logging.info(f"[ASR] Transcript: '{text[:80]}' → lang={lang}")

                await event_callback(
                    ASREvent(
                        type=ASREventType.TRANSCRIPT,
                        transcript=text,
                        language_code=lang,
                    )
                )

            else:
                logging.debug(f"[ASR] Unhandled message type='{msg_type}' signal='{signal_type}'")

        logging.info(f"[ASR] Recv loop exited after {messages_received} total messages.")