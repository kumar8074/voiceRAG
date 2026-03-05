# ===================================================================================
# Project: VoiceRAG
# File: evals/background_eval.py
# Description: Background eval trigger for production use.
#              Called after _save_turn() in agentic_rag.py — zero user latency impact.
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import asyncio
from src.logger import logging
from .schemas import EvalInput
from .eval_runner import EvalRunner

# Module-level singleton — built once, reused across all background tasks
_runner: EvalRunner | None = None


def _get_runner() -> EvalRunner:
    """
    Lazily initialise the EvalRunner singleton.
    Reuses SearchService + RetrievalJudge across all background eval calls
    """
    global _runner
    if _runner is None:
        _runner = EvalRunner()
        logging.info("[BackgroundEval] EvalRunner singleton initialised.")
    return _runner


async def _run_eval_safe(query: str, user_id: str, session_id: str, top_k: int) -> None:
    """
    Internal coroutine that runs the eval pipeline and swallows all exceptions.
    Any failure is logged as a warning — never allowed to propagate and
    never affects the user-facing response or app stability.

    Args:
        query      : The document query that was just answered.
        user_id    : User ID — passed through for Qdrant tenant isolation.
        session_id : Session ID — passed through for Qdrant tenant isolation.
        top_k      : Number of chunks to retrieve for eval (mirrors RAG top_k).
    """
    try:
        logging.info(
            f"[BackgroundEval] Starting eval for query='{query}' "
            f"user={user_id} session={session_id}"
        )

        runner = _get_runner()

        eval_input = EvalInput(
            query      = query,
            user_id    = user_id,
            session_id = session_id,
            top_k      = top_k,
        )

        await runner.evaluate_batch([eval_input])

        logging.info(
            f"[BackgroundEval] Eval completed for query='{query}'"
        )

    except Exception as e:
        # Intentionally broad — eval must never crash the application
        logging.warning(
            f"[BackgroundEval] Eval failed silently for query='{query}': {e}"
        )


def fire_eval(
    query: str,
    user_id: str,
    session_id: str,
    top_k: int = 5,
) -> None:
    """
    Public interface — schedules a background eval task on the running event loop.

    Designed to be called from agentic_rag.py inside _save_turn(), after the
    DB commit, for document queries only. Returns immediately — zero blocking.

    Args:
        query      : The document query that was just answered.
        user_id    : User ID for Qdrant tenant isolation.
        session_id : Session ID for Qdrant tenant isolation.
        top_k      : Chunks to retrieve during eval (default: 5).
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(
            _run_eval_safe(
                query      = query,
                user_id    = user_id,
                session_id = session_id,
                top_k      = top_k,
            ),
            name=f"bg_eval::{user_id}::{session_id}",
        )
        logging.info(
            f"[BackgroundEval] Task scheduled for query='{query}'"
        )

    except RuntimeError as e:
        # No running event loop — should never happen in FastAPI context,
        # but log and skip rather than crash.
        logging.warning(
            f"[BackgroundEval] Could not schedule eval task "
            f"(no running event loop): {e}"
        )