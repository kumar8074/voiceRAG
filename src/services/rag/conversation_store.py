# ===================================================================================
# Project: VoiceRAG
# File: src/services/rag/conversation_store.py
# Description: Provides Conversation Persistence 
# Author: LALAN KUMAR
# Created: [04-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from typing import List, Dict
from sqlmodel import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...models.conversation import ConversationMessage
from ...logger import logging


class ConversationStore:

    async def add_message(
        self,
        session: AsyncSession,
        user_id: str,
        session_id: str,
        role: str,
        message: str
    ) -> None:

        msg = ConversationMessage(
            user_id=user_id,
            session_id=session_id,
            role=role,
            message=message
        )
        session.add(msg)
        await session.flush()
        logging.info(f"Message saved: user={user_id}, session={session_id}, role={role}")

    async def get_history(
        self,
        session: AsyncSession,
        user_id: str,
        session_id: str,
        limit: int = 6
    ) -> List[Dict]:

        statement = (
            select(ConversationMessage)
            .where(
                ConversationMessage.user_id == user_id,
                ConversationMessage.session_id == session_id
            )
            .order_by(ConversationMessage.created_at.desc())
            .limit(limit)
        )

        result = await session.execute(statement)
        rows = result.scalars().all()

        # Reverse to get chronological order
        history = [
            {"role": row.role, "content": row.message}
            for row in reversed(rows)
        ]

        logging.info(f"Fetched {len(history)} history messages for user={user_id}, session={session_id}")
        return history