# ===================================================================================
# Project: VoiceRAG
# File: src/services/voice/tts_handler.py
# Description: Sarvam bulbul:v3 streaming TTS WebSocket lifecycle
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator

from sarvamai import AsyncSarvamAI, AudioOutput, EventResponse

from ...logger import logging
from .audio_utils import decode_mp3_chunk, clamp_language

SARVAM_API_KEY  = os.getenv("SARVAM_API_KEY", "")
TTS_MODEL       = "bulbul:v3"

SUPPORTED_LANGS = {"hi-IN", "en-IN"}
FALLBACK_LANG   = "en-IN"

# Speaker per language — best natural-sounding voices for each
_SPEAKER_MAP: dict[str, str] = {
    "hi-IN": "shubh",   # male, native Hindi
    "en-IN": "arya",    # female, Indian-accented English
}
_DEFAULT_SPEAKER = "arya"


async def synthesize_stream(
    text:          str,
    language_code: str,
    cancel_event:  asyncio.Event,
) -> AsyncGenerator[bytes, None]:
    """
    Stream TTS audio for the given text.

    Args:
        text:          Text to synthesize (full LLM response, correct language).
        language_code: hi-IN or en-IN — clamped to supported set internally.
        cancel_event:  Set this to abort mid-stream (barge-in).

    Yields:
        Raw MP3 bytes, chunk by chunk, as they arrive from Sarvam.

    Raises:
        RuntimeError on connection / synthesis failure (unless barge-in).
    """
    if not text or not text.strip():
        logging.warning("[TTS] Empty text — nothing to synthesize.")
        return

    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set.")

    language_code = clamp_language(language_code, SUPPORTED_LANGS, FALLBACK_LANG)
    speaker       = _SPEAKER_MAP.get(language_code, _DEFAULT_SPEAKER)

    logging.info(
        f"[TTS] Synthesizing {len(text)} chars | "
        f"lang={language_code} speaker={speaker}"
    )

    client = AsyncSarvamAI(api_subscription_key=SARVAM_API_KEY)

    try:
        async with client.text_to_speech_streaming.connect(
            model=TTS_MODEL,
            send_completion_event=True,   # receive "final" event for clean exit
        ) as tts_ws:

            await tts_ws.configure(
                target_language_code=language_code,
                speaker=speaker,
                output_audio_codec="mp3",   # browser-native
                min_buffer_size=50,         # flush after 50 chars for low latency
                max_chunk_length=200,
            )
            logging.info("[TTS] WebSocket configured.")

            await tts_ws.convert(text)
            await tts_ws.flush()
            logging.info("[TTS] Text sent and buffer flushed.")

            chunk_n = 0
            async for message in tts_ws:

                # Barge-in check — abort immediately 
                if cancel_event.is_set():
                    logging.info("[TTS] Barge-in — aborting TTS stream.")
                    return

                if isinstance(message, AudioOutput):
                    chunk_n += 1
                    raw = decode_mp3_chunk(message.data.audio)
                    logging.debug(f"[TTS] Chunk #{chunk_n}: {len(raw)} bytes")
                    yield raw

                elif isinstance(message, EventResponse):
                    if message.data.event_type == "final":
                        logging.info(
                            f"[TTS] Done — {chunk_n} chunks streamed."
                        )
                        return

    except Exception as exc:
        if not cancel_event.is_set():
            # Only raise if this wasn't a barge-in abort
            logging.error(f"[TTS] Error: {exc}")
            raise RuntimeError(f"Sarvam TTS error: {exc}") from exc