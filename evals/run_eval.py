# ===================================================================================
# Project: VoiceRAG
# File: evals/run_eval.py
# Description: CLI entry point for LLM-as-Judge retrieval evaluation
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import asyncio
import argparse
import sys
from pathlib import Path

from src.logger import logging
from .schemas import EvalInput
from .eval_runner import EvalRunner


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m evals.run_eval",
        description="VoiceRAG — LLM-as-Judge Retrieval Evaluator",
    )

    # Query source — exactly one of these must be provided
    query_group = parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument(
        "--query",
        type=str,
        help="A single query string to evaluate.",
    )
    query_group.add_argument(
        "--queries-file",
        type=str,
        metavar="PATH",
        help="Path to a .txt file with one query per line.",
    )

    # Required tenant identifiers (mirrors SearchService.search signature)
    parser.add_argument(
        "--user-id",
        type=str,
        required=True,
        help="User ID for Qdrant tenant isolation.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        required=True,
        help="Session ID for Qdrant tenant isolation.",
    )

    # Optional
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        metavar="N",
        help="Number of chunks to retrieve per query (default: 5).",
    )

    return parser.parse_args()


def _load_queries(args: argparse.Namespace) -> list[str]:
    """
    Return a list of query strings from CLI args.
    Handles both --query (single) and --queries-file (batch).
    """
    if args.query:
        queries = [args.query.strip()]
        logging.info(f"[run_eval] Single query mode: '{queries[0]}'")
        return queries

    queries_path = Path(args.queries_file)
    if not queries_path.exists():
        logging.error(f"[run_eval] Queries file not found: {queries_path}")
        sys.exit(1)

    raw_lines = queries_path.read_text(encoding="utf-8").splitlines()
    queries   = [line.strip() for line in raw_lines if line.strip()]

    if not queries:
        logging.error(f"[run_eval] Queries file is empty: {queries_path}")
        sys.exit(1)

    logging.info(
        f"[run_eval] Batch mode: loaded {len(queries)} query/queries "
        f"from '{queries_path}'"
    )
    return queries


async def main() -> None:
    args    = _parse_args()
    queries = _load_queries(args)

    eval_inputs = [
        EvalInput(
            query      = q,
            user_id    = args.user_id,
            session_id = args.session_id,
            top_k      = args.top_k,
        )
        for q in queries
    ]

    logging.info(
        f"[run_eval] Starting eval — "
        f"{len(eval_inputs)} query/queries | top_k={args.top_k} | "
        f"user_id={args.user_id} | session_id={args.session_id}"
    )

    runner = EvalRunner()
    report = await runner.evaluate_batch(eval_inputs)

    logging.info(
        f"[run_eval] Eval complete — "
        f"overall_avg={report.avg_overall_score:.4f} | "
        f"PASS={report.passed} PARTIAL={report.partial} FAIL={report.failed}"
    )


if __name__ == "__main__":
    asyncio.run(main())