# ===================================================================================
# Project: VoiceRAG
# File: evals/eval_runner.py
# Description: Orchestrates retrieval eval — search → judge → EvalReport
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import json
from datetime import datetime
from pathlib import Path
from typing import List

from src.logger import logging
from src.services.search.search_service import SearchService
from .judges import RetrievalJudge
from .schemas import EvalInput, EvalResult, EvalReport

# Output directories
EVAL_OUTPUT_DIR = Path("evals/reports")


class EvalRunner:
    """
    Orchestrates the full LLM-as-Judge evaluation pipeline for VoiceRAG.

    For each EvalInput:
        1. Calls SearchService.search() to retrieve top-K chunks
        2. Passes each chunk to RetrievalJudge for LLM scoring
        3. Aggregates scores into an EvalResult with verdict

    For a batch of EvalInputs:
        4. Collects all EvalResults into an EvalReport
        5. Saves report as JSON  → evals/reports/<timestamp>.json
        6. Saves report as Markdown → evals/reports/<timestamp>.md
        7. Pretty-prints summary to console
    """

    def __init__(self):
        self.searcher = SearchService()
        self.judge    = RetrievalJudge()
        EVAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        logging.info("[EvalRunner] Initialised.")

    # Single query evaluation
    async def evaluate_one(self, eval_input: EvalInput) -> EvalResult:
        """
        Run the full eval pipeline for a single query.

        Args:
            eval_input: EvalInput with query, user_id, session_id, top_k.

        Returns:
            EvalResult with per-chunk scores and aggregate verdict.
        """
        logging.info(
            f"[EvalRunner] Evaluating query: '{eval_input.query}' "
            f"| user={eval_input.user_id} | session={eval_input.session_id}"
        )

        # Retrieve chunks
        chunks = await self.searcher.search(
            query      = eval_input.query,
            user_id    = eval_input.user_id,
            session_id = eval_input.session_id,
            top_k      = eval_input.top_k,
        )
        logging.info(f"[EvalRunner] Retrieved {len(chunks)} chunk(s).")

        if not chunks:
            logging.warning(
                f"[EvalRunner] No chunks retrieved for query: '{eval_input.query}'. "
                f"Returning FAIL result."
            )
            return EvalResult(
                query                  = eval_input.query,
                user_id                = eval_input.user_id,
                session_id             = eval_input.session_id,
                total_chunks_retrieved = 0,
                chunk_scores           = [],
                avg_context_relevance  = 0.0,
                avg_context_coverage   = 0.0,
                avg_faithfulness       = 0.0,
                overall_score          = 0.0,
                retrieval_verdict      = "FAIL",
            )

        # Judge each chunk
        chunk_scores = await self.judge.score_all_chunks(
            query  = eval_input.query,
            chunks = chunks,
        )

        # Aggregate
        agg = self.judge.aggregate(chunk_scores)

        result = EvalResult(
            query                  = eval_input.query,
            user_id                = eval_input.user_id,
            session_id             = eval_input.session_id,
            total_chunks_retrieved = len(chunks),
            chunk_scores           = chunk_scores,
            avg_context_relevance  = agg["avg_context_relevance"],
            avg_context_coverage   = agg["avg_context_coverage"],
            avg_faithfulness       = agg["avg_faithfulness"],
            overall_score          = agg["overall_score"],
            retrieval_verdict      = agg["retrieval_verdict"],
        )

        logging.info(
            f"[EvalRunner] Result for '{eval_input.query}': "
            f"overall={result.overall_score} | verdict={result.retrieval_verdict}"
        )
        return result

    # Batch evaluation
    async def evaluate_batch(self, eval_inputs: List[EvalInput]) -> EvalReport:
        """
        Evaluate a list of queries and produce a full EvalReport.

        Args:
            eval_inputs: List of EvalInput objects.

        Returns:
            EvalReport with per-query results and batch-level aggregates.
        """
        if not eval_inputs:
            raise ValueError("eval_inputs cannot be empty.")

        logging.info(f"[EvalRunner] Starting batch eval for {len(eval_inputs)} query/queries.")

        results: List[EvalResult] = []
        for idx, eval_input in enumerate(eval_inputs, start=1):
            logging.info(f"[EvalRunner] Query {idx}/{len(eval_inputs)} ...")
            result = await self.evaluate_one(eval_input)
            results.append(result)

        # Batch aggregates
        n             = len(results)
        passed        = sum(1 for r in results if r.retrieval_verdict == "PASS")
        partial       = sum(1 for r in results if r.retrieval_verdict == "PARTIAL")
        failed        = sum(1 for r in results if r.retrieval_verdict == "FAIL")

        avg_relevance = round(sum(r.avg_context_relevance for r in results) / n, 4)
        avg_coverage  = round(sum(r.avg_context_coverage  for r in results) / n, 4)
        avg_faith     = round(sum(r.avg_faithfulness       for r in results) / n, 4)
        avg_overall   = round(sum(r.overall_score          for r in results) / n, 4)

        report = EvalReport(
            total_queries          = n,
            passed                 = passed,
            partial                = partial,
            failed                 = failed,
            avg_context_relevance  = avg_relevance,
            avg_context_coverage   = avg_coverage,
            avg_faithfulness       = avg_faith,
            avg_overall_score      = avg_overall,
            results                = results,
        )

        logging.info(
            f"[EvalRunner] Batch complete — "
            f"PASS={passed}, PARTIAL={partial}, FAIL={failed}, "
            f"avg_overall={avg_overall}"
        )

        # Persist outputs
        timestamp  = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        json_path  = EVAL_OUTPUT_DIR / f"eval_{timestamp}.json"
        md_path    = EVAL_OUTPUT_DIR / f"eval_{timestamp}.md"

        self._save_json(report, json_path)
        self._save_markdown(report, md_path)
        self._log_summary(report)

        return report

    @staticmethod
    def _save_json(report: EvalReport, path: Path) -> None:
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        logging.info(f"[EvalRunner] JSON report saved → {path}")

    @staticmethod
    def _save_markdown(report: EvalReport, path: Path) -> None:
        lines = [
            "# VoiceRAG — Retrieval Eval Report",
            f"\n**Generated at:** {report.generated_at}  ",
            f"**Total queries:** {report.total_queries}  ",
            "",
            "## Batch Summary",
            "",
            "| Metric | Score |",
            "|--------|-------|",
            f"| ✅ Passed       | {report.passed} / {report.total_queries} |",
            f"| ⚠️ Partial      | {report.partial} / {report.total_queries} |",
            f"| ❌ Failed       | {report.failed} / {report.total_queries} |",
            f"| Avg Relevance  | {report.avg_context_relevance:.4f} |",
            f"| Avg Coverage   | {report.avg_context_coverage:.4f} |",
            f"| Avg Faithfulness | {report.avg_faithfulness:.4f} |",
            f"| **Avg Overall** | **{report.avg_overall_score:.4f}** |",
            "",
            "---",
            "",
            "## Per-Query Results",
            "",
        ]

        for idx, result in enumerate(report.results, start=1):
            verdict_emoji = {"PASS": "✅", "PARTIAL": "⚠️", "FAIL": "❌"}.get(
                result.retrieval_verdict, "❓"
            )
            lines += [
                f"### {idx}. {result.query}",
                "",
                f"**Verdict:** {verdict_emoji} {result.retrieval_verdict}  ",
                f"**Overall Score:** {result.overall_score:.4f}  ",
                f"**Chunks Retrieved:** {result.total_chunks_retrieved}  ",
                "",
                "| Dimension | Score |",
                "|-----------|-------|",
                f"| Context Relevance | {result.avg_context_relevance:.4f} |",
                f"| Context Coverage  | {result.avg_context_coverage:.4f} |",
                f"| Faithfulness      | {result.avg_faithfulness:.4f} |",
                "",
                "**Chunk-level breakdown:**",
                "",
            ]

            for cs in result.chunk_scores:
                lines += [
                    f"- **Chunk {cs.chunk_index + 1}** "
                    f"(qdrant_score={cs.retrieval_score:.3f}) — "
                    f"relevance={cs.context_relevance}, "
                    f"coverage={cs.context_coverage}, "
                    f"faithfulness={cs.faithfulness}  ",
                    f"  > *{cs.judge_reasoning}*  ",
                    f"  > Preview: `{cs.content_preview[:120]}...`",
                    "",
                ]

            lines.append("---\n")

        path.write_text("\n".join(lines), encoding="utf-8")
        logging.info(f"[EvalRunner] Markdown report saved → {path}")

    @staticmethod
    def _log_summary(report: EvalReport) -> None:
        sep = "=" * 55
        logging.info(sep)
        logging.info("VoiceRAG Retrieval Eval — Summary")
        logging.info(sep)
        logging.info(f"Total queries    : {report.total_queries}")
        logging.info(f"PASS             : {report.passed}")
        logging.info(f"PARTIAL          : {report.partial}")
        logging.info(f"FAIL             : {report.failed}")
        logging.info(sep)
        logging.info(f"Avg Relevance    : {report.avg_context_relevance:.4f}")
        logging.info(f"Avg Coverage     : {report.avg_context_coverage:.4f}")
        logging.info(f"Avg Faithfulness : {report.avg_faithfulness:.4f}")
        logging.info(f"Avg Overall      : {report.avg_overall_score:.4f}")
        logging.info(sep)

        for idx, r in enumerate(report.results, start=1):
            logging.info(
                f"[{idx:02d}] verdict={r.retrieval_verdict:<7} "
                f"overall={r.overall_score:.4f}  |  query='{r.query[:55]}'"
            )