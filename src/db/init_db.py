# ===================================================================================
# Project: VoiceRAG
# File: src/db/init_db.py
# Description: DB initialization interface, used inside lifespan
# Author: LALAN KUMAR
# Created: [04-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from sqlmodel import SQLModel
from .factory import engine

from ..models.conversation import ConversationMessage

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(SQLModel.metadata.create_all)
