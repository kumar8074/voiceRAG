from sqlmodel import SQLModel, Field
from datetime import datetime, timezone
from typing import Optional, Text

class ConversationMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    
    user_id: str = Field(index=True)
    session_id: str = Field(index=True)
    
    role: str = Field(index=True)
    message: str
    
    created_at: datetime = Field(default_factory=lambda: datetime.utcnow())
    