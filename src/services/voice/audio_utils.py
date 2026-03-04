# ===================================================================================
# Project: VoiceRAG
# File: src/services/voice/audio_utils.py
# Description: Pure audio utility helpers — no I/O, no external dependencies.
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import base64
import re

def pcm_to_base64(pcm_bytes: bytes) -> str:
    """Encode raw PCM bytes (s16le, 16kHz) to base64 for Sarvam ASR."""
    return base64.b64encode(pcm_bytes).decode("utf-8")


def decode_mp3_chunk(b64_audio: str) -> bytes:
    """Decode a base64-encoded MP3 chunk from Sarvam TTS to raw bytes."""
    return base64.b64decode(b64_audio)


# Devanagari Unicode block: U+0900–U+097F
# Any transcript containing these characters is Hindi.
_DEVANAGARI_RE = re.compile(r'[\u0900-\u097F]')


def detect_language(text: str) -> str:
    """
    Detect whether a transcript is Hindi or English using script detection.

    This is more reliable than relying on Sarvam's language_code field,
    which is not returned for saaras:v3 STT mode (only for STTT/translate).

    Rules:
        - Any Devanagari character present → hi-IN
        - Otherwise                        → en-IN

    Args:
        text: Transcribed text from Sarvam ASR.

    Returns:
        "hi-IN" or "en-IN"
    """
    if _DEVANAGARI_RE.search(text):
        return "hi-IN"
    return "en-IN"


def clamp_language(language_code: str, supported: set, fallback: str) -> str:
    """
    Return language_code if supported, otherwise return fallback.
    Single source of truth for language filtering across all modules.
    """
    return language_code if language_code in supported else fallback