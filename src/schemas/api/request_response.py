from pydantic import BaseModel, Field
from typing import Literal

class ChatRequest(BaseModel):
    query: str
    user_id: str
    session_id: str
    top_k: int = 5
    
class ASRResponse(BaseModel):
    """Returned by POST /api/v1/voice/asr"""
    transcript:    str  = Field(..., description="Transcribed text.")
    language_code: Literal["hi-IN", "en-IN"] = Field(
        ...,
        description="Detected language (only hi-IN and en-IN are supported)."
    )
    
class TTSRequest(BaseModel):
    """Body for POST /api/v1/voice/tts"""
    text: str = Field(..., description="Text to synthesize.")
    language_code: Literal["hi-IN", "en-IN"] = Field(
        "en-IN",
        description="Language for synthesis — determines speaker voice."
    )