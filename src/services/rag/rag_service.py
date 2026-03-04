# ===================================================================================
# Project: VoiceRAG
# File: src/services/rag/rag_service.py
# Description: RAG Service Layer
# Author: LALAN KUMAR
# Created: [04-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import re
import asyncio
from typing import AsyncGenerator, List, Dict
from sqlalchemy.ext.asyncio import AsyncSession

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from ...config import LLM_MODEL
from ...logger import logging
from ..search.search_service import SearchService
from .conversation_store import ConversationStore
from .prompt_builder import PromptBuilder

GENERAL_PATTERNS = re.compile(
    r"""
    ^\s*(
        # English
        hi+|hello+|hey+|howdy|
        how\s+are\s+you|how\s+r\s+u|what'?s?\s+up|
        good\s+(morning|evening|afternoon|night|day)|
        bye|goodbye|see\s+you|take\s+care|
        thanks?|thank\s+you|thx|ty|
        ok+|okay|sure|cool|got\s+it|great|nice|
        who\s+are\s+you|what\s+are\s+you|
        # Hindi / Hinglish
        namaste|namaskar|pranam|
        kaise\s+ho|kya\s+haal|kya\s+chal|sab\s+theek|
        shukriya|dhanyawad|dhanyavaad|
        acha|accha|theek\s+hai|bilkul|haan|nahi|
        alvida|phir\s+milenge|
        # Tamil
        vanakkam|romba\s+nandri|nandri|sari|
        # Telugu
        namaskaram|dhanyavaadalu|baagunnara|
        # Bengali
        nomoshkar|dhonnobad|kemon\s+acho|
        # Marathi
        namaskar|dhanyawad|kasa\s+ahat|
        # Kannada
        namaskara|dhanyavadagalu|hegiddira
    )\s*[!?.]*\s*$
    """,
    re.VERBOSE | re.IGNORECASE
)

INTENT_CLASSIFIER_SYSTEM = """You are an intent classifier.
Reply with ONLY a single digit — no explanation, no punctuation, nothing else:
0 = general conversation (greetings, small talk, chit-chat, how are you, thanks, bye, casual messages in ANY language or mix of languages)
1 = needs document search (questions about specific topics, documents, facts, data, information requests in ANY language)"""

GENERAL_SYSTEM_PROMPT = """You are a friendly, helpful voice assistant.
Respond naturally and concisely.
Respond in the SAME language as the user's message.
"""

class RAGService:
    """
    Core RAG orchestrator for VoiceRAG.

    Flow (2 Layer Intent classification):
        1a. Classify intent instantly via regex (zero latency)
        1b. LLM Classifier with max token 1, (150-200ms latency)
        2a. General  → stream LLM response directly
        2b. Document → fetch history + vector search IN PARALLEL
                     → build prompt → stream LLM response
        3. Persist conversation turn to DB AFTER stream completes (background)
    """

    def __init__(self):
        self.llm = LLM_MODEL
        self.searcher = SearchService()
        self.store = ConversationStore()

    async def _is_general_query(self, query: str) -> bool:
        """
        Layer 1: regex for greetings
        Layer 2: For edge cases
        """
        # Layer 1
        if GENERAL_PATTERNS.match(query):
            logging.info("Intent resolved by regex -> General")
            return True
        
        query_str=str(query)
        # Layer 2
        response = await self.llm.ainvoke(
            [
                SystemMessage(content=INTENT_CLASSIFIER_SYSTEM),
                HumanMessage(content=query_str)
            ],
            config={"max_tokens":1}
        )
        intent = response.content.strip()
        logging.info(f"Intent resolved by LLM classifier → '{intent}' for: '{query_str}'")
        if intent == "0":
            return True   # general
        if intent == "1":
            return False 
        
        logging.warning(f"Unexpected intent classifier response: '{intent}', defaulting to document")
        return False


    def _build_message_history(self, history: List[Dict]) -> List:
        messages = []
        for turn in history:
            if turn["role"] == "user":
                messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                messages.append(AIMessage(content=turn["content"]))
        return messages


    async def query(
        self,
        query: str,
        user_id: str,
        session_id: str,
        db: AsyncSession,
        top_k: int = 5
    ) -> AsyncGenerator[str, None]:
        """
        Main entry point. Yields streamed tokens as they arrive from LLM.
        DB persistence happens after stream ends — no latency added.
        """

        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        is_general = await self._is_general_query(query)
        logging.info(f"Intent: {'general' if is_general else 'document'} — query: '{query}'")

        if is_general:
            async for token in self._handle_general(
                query=query,
                user_id=user_id,
                session_id=session_id,
                db=db
            ):
                yield token
        else:
            async for token in self._handle_document(
                query=query,
                user_id=user_id,
                session_id=session_id,
                db=db,
                top_k=top_k
            ):
                yield token

    async def _handle_general(
        self,
        query: str,
        user_id: str,
        session_id: str,
        db: AsyncSession
    ) -> AsyncGenerator[str, None]:

        messages = [
            SystemMessage(content=GENERAL_SYSTEM_PROMPT),
            HumanMessage(content=query)
        ]

        collected = []

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                collected.append(chunk.content)
                yield chunk.content          # ← user sees token immediately

        # Persist AFTER stream ends — zero impact on perceived latency
        full_response = "".join(collected)
        await self._save_turn(db, user_id, session_id, query, full_response)

    async def _handle_document(
        self,
        query: str,
        user_id: str,
        session_id: str,
        db: AsyncSession,
        top_k: int
    ) -> AsyncGenerator[str, None]:

        # Parallel fetch: history + vector search
        logging.info("Fetching history and search results in parallel...")
        history, contexts = await asyncio.gather(
            self.store.get_history(
                session=db,
                user_id=user_id,
                session_id=session_id
            ),
            self.searcher.search(
                query=query,
                user_id=user_id,
                session_id=session_id,
                top_k=top_k
            )
        )
        logging.info(f"Parallel fetch done — {len(history)} history turns, {len(contexts)} chunks")

        # Build prompt
        rag_prompt = PromptBuilder.build_prompt(
            query=query,
            contexts=contexts,
            history=history
        )

        messages = self._build_message_history(history) + [
            HumanMessage(content=rag_prompt)
        ]

        collected = []

        async for chunk in self.llm.astream(messages):
            if chunk.content:
                collected.append(chunk.content)
                yield chunk.content          # ← user sees token immediately

        # Persist AFTER stream ends
        full_response = "".join(collected)
        await self._save_turn(db, user_id, session_id, query, full_response)

    async def _save_turn(
        self,
        db: AsyncSession,
        user_id: str,
        session_id: str,
        query: str,
        response: str
    ) -> None:
        await self.store.add_message(db, user_id, session_id, "user", query),
        await self.store.add_message(db, user_id, session_id, "assistant", response)
        await db.commit() 
        
        logging.info(f"Turn persisted: user={user_id}, session={session_id}")
        
        
# Example usage:
if __name__ == "__main__":
    import asyncio
    from src.db.factory import async_session
    from src.db.init_db import init_db

    async def run_query(rag: RAGService, db, query: str, user_id: str, session_id: str):
        print(f"\n{'─'*60}")
        print(f"USER : {query}")
        print(f"BOT  : ", end="", flush=True)

        async for token in rag.query(
            query=query,
            user_id=user_id,
            session_id=session_id,
            db=db,
            top_k=5
        ):
            print(token, end="", flush=True)

        print()

    async def main():
        await init_db()
        print("DB ready.\n")

        rag = RAGService()

        async with async_session() as db:
            user_id    = "test_user_1"
            session_id = "test_session_1"

            # General queries — should NOT touch Qdrant
            await run_query(rag, db, "Hi!",          user_id, session_id)
            await run_query(rag, db, "Namaste",       user_id, session_id)
            await run_query(rag, db, "Thanks!",       user_id, session_id)

            # Document queries — should hit Qdrant + history in parallel
            await run_query(rag, db, "What is Startup India initiative?",  user_id, session_id)
            await run_query(rag, db, "Startup India ke kya fayde hain?",   user_id, session_id)
            await run_query(rag, db, "What are the eligibility criteria?", user_id, session_id)

    asyncio.run(main())