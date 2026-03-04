# ===================================================================================
# Project: VoiceRAG
# File: src/models/conversation.py
# Description: DB Schema
# Author: LALAN KUMAR
# Created: [04-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from sqlmodel import SQLModel, Field
from datetime import datetime
from typing import Optional

class ConversationMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    
    user_id: str = Field(index=True)
    session_id: str = Field(index=True)
    
    role: str = Field(index=True)
    message: str
    
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    