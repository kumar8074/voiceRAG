# ===================================================================================
# Project: VoiceRAG
# File: src/services/rag/agentic_rag.py
# Description: Agentic RAG using LangGraph
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import re
import json
import asyncio
import os
from dotenv import load_dotenv
from typing import AsyncGenerator, Dict
from sqlalchemy.ext.asyncio import AsyncSession

from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from evals.background_eval import fire_eval

from ...config import LLM_MODEL
from ...logger import logging
from ..search.search_service import SearchService
from .conversation_store import ConversationStore
from .states import RAGState
from .prompts import (INTENT_CLASSIFIER_SYSTEM, GENERAL_SYSTEM_PROMPT,
                      RELEVANCE_GRADER_SYSTEM, RAG_SYSTEM_PROMPT,
                      QUERY_DECOMPOSER_SYSTEM, NO_RESULTS_SYSTEM)
from .prompt_builder import PromptBuilder

load_dotenv()

os.environ["LANGSMITH_TRACING"] = os.getenv("LANGSMITH_TRACING", "false")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "voicerag")

MAX_RETRIES = 2   # max rewrite+search loops → 3 searches total

GENERAL_PATTERNS = re.compile(
    r"""
    ^\s*(
        hi+|hello+|hey+|howdy|
        how\s+are\s+you|how\s+r\s+u|what'?s?\s+up|
        good\s+(morning|evening|afternoon|night|day)|
        bye|goodbye|see\s+you|take\s+care|
        thanks?|thank\s+you|thx|ty|
        ok+|okay|sure|cool|got\s+it|great|nice|
        who\s+are\s+you|what\s+are\s+you|
        namaste|namaskar|pranam|
        kaise\s+ho|kya\s+haal|kya\s+chal|sab\s+theek|
        shukriya|dhanyawad|dhanyavaad|
        acha|accha|theek\s+hai|bilkul|haan|nahi|
        alvida|phir\s+milenge|
        vanakkam|romba\s+nandri|nandri|sari|
        namaskaram|dhanyavaadalu|baagunnara|
        nomoshkar|dhonnobad|kemon\s+acho|
        namaskar|dhanyawad|kasa\s+ahat|
        namaskara|dhanyavadagalu|hegiddira
    )\s*[!?.]*\s*$
    """,
    re.VERBOSE | re.IGNORECASE
)

class RAGService:
    """
    The compiled graph is built once at __init__ and reused across all
    requests. 
    """

    def __init__(self):
        self.llm      = LLM_MODEL
        self.searcher = SearchService()
        self.store    = ConversationStore()
        self._graph   = self._build_graph()

    # Graph builder
    def _build_graph(self):
        g = StateGraph(RAGState)

        g.add_node("classify_intent",    self._node_classify_intent)
        g.add_node("general_response",   self._node_general_response)
        g.add_node("fetch_history",      self._node_fetch_history)
        g.add_node("search",             self._node_search)
        g.add_node("verify_relevance",   self._node_verify_relevance)
        g.add_node("rewrite_query",      self._node_rewrite_query)
        g.add_node("generate",           self._node_generate)
        g.add_node("handle_no_results",  self._node_handle_no_results)

        # Fixed edges
        g.add_edge(START,              "classify_intent")
        g.add_edge("fetch_history",    "search")
        g.add_edge("search", "verify_relevance")
        g.add_edge("rewrite_query",    "search")
        g.add_edge("general_response", END)
        g.add_edge("generate",         END)
        g.add_edge("handle_no_results", END)

        # Conditional: classify_intent → general_response | fetch_history
        g.add_conditional_edges(
            "classify_intent",
            lambda s: "general_response" if s["is_general"] else "fetch_history"
        )

        # Conditional: verify_relevance → generate | rewrite_query | handle_no_results
        g.add_conditional_edges(
            "verify_relevance",
            self._route_after_verify
        )

        return g.compile()

    def _route_after_verify(self, state: RAGState) -> str:
        if state["contexts"]:
            return "generate"
        if state["retry_count"] < MAX_RETRIES:
            return "rewrite_query"
        return "handle_no_results"

    # Nodes
    async def _node_classify_intent(self, state: RAGState) -> dict:
        query = state["query"]

        if GENERAL_PATTERNS.match(query):
            logging.info("[Graph] Intent resolved by regex → general")
            return {"is_general": True, "active_query": query, "retry_count": 0}

        response = await self.llm.ainvoke(
            [
                SystemMessage(content=INTENT_CLASSIFIER_SYSTEM),
                HumanMessage(content=str(query))
            ],
            config={"max_tokens": 1}
        )
        intent = response.content.strip()
        is_general = (intent == "0")
        logging.info(f"[Graph] Intent resolved by LLM → '{intent}' for: '{query}'")
        return {"is_general": is_general, "active_query": query, "retry_count": 0}


    async def _node_general_response(self, state: RAGState) -> dict:
        messages = [
            SystemMessage(content=GENERAL_SYSTEM_PROMPT),
            HumanMessage(content=state["query"])
        ]
        response = await self.llm.ainvoke(messages)
        full_response = response.content.strip()

        await self._save_turn(
            state["db"], state["user_id"], state["session_id"],
            state["query"], full_response,
            is_document_query=False
        )
        logging.info(f"[Graph] General response: {len(full_response)} chars.")
        return {"final_response": full_response}


    async def _node_fetch_history(self, state: RAGState) -> dict:
        history = await self.store.get_history(
            session=state["db"],
            user_id=state["user_id"],
            session_id=state["session_id"],
        )
        logging.info(f"[Graph] Fetched {len(history)} history turns.")
        return {"history": history}


    async def _node_search(self, state: RAGState) -> dict:
        """
        Attempt 1:  single search with the original query.
        Attempts 2+: active_query is a JSON array of sub-queries from
                     rewrite_query — all run concurrently, results merged
                     and deduplicated by chunk_id.
        """
        active_query = state["active_query"]
        retry_count  = state["retry_count"]

        if retry_count == 0:
            queries = [active_query]
        else:
            try:
                parsed = json.loads(active_query)
                queries = parsed if isinstance(parsed, list) and parsed else [active_query]
            except json.JSONDecodeError:
                queries = [active_query]

        logging.info(
            f"[Graph] Search attempt {retry_count + 1}/{MAX_RETRIES + 1} "
            f"with {len(queries)} sub-query/queries."
        )

        tasks = [
            self.searcher.search(
                query=q,
                user_id=state["user_id"],
                session_id=state["session_id"],
                top_k=5
            )
            for q in queries
        ]
        results_per_query = await asyncio.gather(*tasks)

        # Merge, deduplicate by chunk_id, keep highest score, cap at 5
        seen: Dict[str, Dict] = {}
        for results in results_per_query:
            for r in results:
                cid = r["chunk_id"]
                if cid not in seen or r["score"] > seen[cid]["score"]:
                    seen[cid] = r

        merged = sorted(seen.values(), key=lambda x: x["score"], reverse=True)[:5]
        logging.info(f"[Graph] Search returned {len(merged)} unique chunk(s).")
        return {"contexts": merged}


    async def _node_verify_relevance(self, state: RAGState) -> dict:
        contexts = state["contexts"]
        query    = state["query"]

        if not contexts:
            logging.info("[Graph] No chunks returned — relevance: FAIL")
            return {"contexts": []}

        chunks_text = "\n\n".join([
            f"[Chunk {i+1} | score={r['score']:.3f}]\n{r['content']}"
            for i, r in enumerate(contexts)
        ])

        try:
            response = await self.llm.ainvoke([
                SystemMessage(content=RELEVANCE_GRADER_SYSTEM),
                HumanMessage(content=(
                    f"Query: {query}\n\n"
                    f"Retrieved chunks:\n{chunks_text}"
                ))
            ])

            raw = response.content.strip()
            logging.info(f"[Graph] Relevance grader raw response: {raw!r}")

            try:
                parsed   = json.loads(raw)
                relevant = bool(parsed.get("relevant", False))
                reason   = parsed.get("reason", "")
            except (json.JSONDecodeError, AttributeError):
                logging.warning(
                    f"[Graph] Relevance grader returned unparseable: {raw!r} "
                    f"— defaulting to relevant"
                )
                relevant = True
                reason   = "parse error — defaulting to relevant"

            logging.info(
                f"[Graph] Relevance: {'PASS' if relevant else 'FAIL'} "
                f"| retry={state['retry_count']} | reason: {reason}"
            )
            return {"contexts": contexts if relevant else []}

        except Exception as e:
            logging.error(f"[Graph] verify_relevance exception: {e} — defaulting to relevant")
            return {"contexts": contexts}


    async def _node_rewrite_query(self, state: RAGState) -> dict:
        """
        Decomposes the original query into focused sub-queries.
        Returns a JSON array string stored in active_query.
        """
        new_retry = state["retry_count"] + 1
        logging.info(f"[Graph] Rewriting query — attempt {new_retry}/{MAX_RETRIES}")

        response = await self.llm.ainvoke([
            SystemMessage(content=QUERY_DECOMPOSER_SYSTEM),
            HumanMessage(content=(
                f"Original query: {state['query']}\n"
                f"Previous search query: {state['active_query']}\n"
                f"This search returned irrelevant or no results. "
                f"Decompose into focused sub-queries."
            ))
        ])

        raw = response.content.strip()
        try:
            parsed = json.loads(raw)
            if not isinstance(parsed, list) or not parsed:
                raise ValueError("Not a non-empty list")
            active_query = raw
            logging.info(f"[Graph] Sub-queries: {parsed}")
        except (json.JSONDecodeError, ValueError):
            logging.warning(
                f"[Graph] Query decomposer returned malformed JSON: {raw!r} "
                f"— falling back to original query"
            )
            active_query = json.dumps([state["query"]])

        return {"active_query": active_query, "retry_count": new_retry}


    async def _node_generate(self, state: RAGState) -> dict:
        rag_prompt = PromptBuilder.build_prompt(
            query=state["query"],
            contexts=state["contexts"],
            history=state["history"]
        )

        history_messages = []
        for turn in state["history"]:
            if turn["role"] == "user":
                history_messages.append(HumanMessage(content=turn["content"]))
            elif turn["role"] == "assistant":
                history_messages.append(AIMessage(content=turn["content"]))

        messages = (
            [SystemMessage(content=RAG_SYSTEM_PROMPT)]
            + history_messages
            + [HumanMessage(content=rag_prompt)]
        )

        response = await self.llm.ainvoke(messages)
        full_response = response.content.strip()

        await self._save_turn(
            state["db"], state["user_id"], state["session_id"],
            state["query"], full_response,
            is_document_query=True
        )
        logging.info(f"[Graph] Generated response: {len(full_response)} chars.")
        return {"final_response": full_response}


    async def _node_handle_no_results(self, state: RAGState) -> dict:
        logging.info(
            f"[Graph] All {MAX_RETRIES + 1} search attempts exhausted "
            f"— no relevant chunks found."
        )
        response = await self.llm.ainvoke([
            SystemMessage(content=NO_RESULTS_SYSTEM),
            HumanMessage(content=state["query"])
        ])
        full_response = response.content.strip()

        await self._save_turn(
            state["db"], state["user_id"], state["session_id"],
            state["query"], full_response,
            is_document_query=False
        )
        logging.info(f"[Graph] No-results response: {len(full_response)} chars.")
        return {"final_response": full_response}

    # query point
    async def query(
        self,
        query: str,
        user_id: str,
        session_id: str,
        db: AsyncSession,
        top_k: int = 5,
    ) -> AsyncGenerator[str, None]:
        """
        Runs the LangGraph pipeline and streams the final response
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        initial_state: RAGState = {
            "query":          query,
            "user_id":        user_id,
            "session_id":     session_id,
            "db":             db,
            "is_general":     False,
            "history":        [],
            "contexts":       [],
            "active_query":   query,
            "retry_count":    0,
            "final_response": "",
        }

        try:
            final_state    = await self._graph.ainvoke(initial_state)
            final_response = final_state.get("final_response", "").strip()

            if not final_response:
                logging.warning("[Graph] final_response is empty after graph completion.")
                return

            logging.info(
                f"[Graph] Streaming {len(final_response)} chars to client "
                f"for user={user_id}"
            )

            # Yield word-by-word
            words = final_response.split(" ")
            for i, word in enumerate(words):
                token = word if i == len(words) - 1 else word + " "
                yield token
                await asyncio.sleep(0)

        except Exception as e:
            logging.error(f"[Graph] Error during pipeline: {e}")
            raise

    # Persistence helper
    async def _save_turn(
        self,
        db: AsyncSession,
        user_id: str,
        session_id: str,
        query: str,
        response: str,
        is_document_query: bool = False, 
    ) -> None:
        await self.store.add_message(db, user_id, session_id, "user", query)
        await self.store.add_message(db, user_id, session_id, "assistant", response)
        await db.commit()
        logging.info(
            f"[Graph] Turn persisted: user={user_id}, session={session_id}"
        )
        
        # Fire background eval only for document queries
        if is_document_query:
            fire_eval(
                query=query,
                user_id=user_id,
                session_id=session_id
            )


# Example usage
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
            await run_query(rag, db, "Hi!",                        user_id, session_id)
            await run_query(rag, db, "Namaste",                    user_id, session_id)
            await run_query(rag, db, "summarize the document",     user_id, session_id)
            await run_query(rag, db, "what are the key findings?", user_id, session_id)

    asyncio.run(main())