# ===================================================================================
# Project: VoiceRAG
# File: evals/judges.py
# Description: LLM-as-Judge scoring logic for retrieval evaluation
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import json
from typing import List, Dict, Any

from langchain_core.messages import SystemMessage, HumanMessage

from src.config import LLM_MODEL
from src.logger import logging
from .schemas import ChunkScore


# Verdict thresholds
PASS_THRESHOLD    = 0.7
PARTIAL_THRESHOLD = 0.4

# Judge system prompt
CHUNK_JUDGE_SYSTEM = """You are an expert RAG evaluation judge.

Given a user query and a single retrieved document chunk, score the chunk on
three dimensions. Reply with ONLY a valid JSON object — no explanation, no markdown fences:

{
  "context_relevance": <float 0.0–1.0>,
  "context_coverage": <float 0.0–1.0>,
  "faithfulness": <float 0.0–1.0>,
  "reasoning": "<one concise sentence>"
}

Scoring guide:
- context_relevance : How topically relevant is this chunk to the query?
                      0 = completely off-topic, 1 = directly on-topic.
- context_coverage  : How much of what the query asks does this chunk address?
                      0 = addresses nothing, 1 = fully answers the query on its own.
- faithfulness      : Is the chunk internally coherent, well-formed and non-noisy?
                      0 = garbled / corrupt / contradictory, 1 = clean and trustworthy.

Be strict and objective. Do NOT reward verbosity."""

# Judge class
class RetrievalJudge:
    """
    Calls the Sarvam LLM to score each retrieved chunk independently.

    One LLM call per chunk — keeps the prompt small and scores precise.
    All calls are made sequentially to avoid hammering the API rate limit.
    """

    def __init__(self):
        self.llm = LLM_MODEL
        logging.info("[Judge] RetrievalJudge initialised with Sarvam LLM.")

    async def score_chunk(
        self,
        query: str,
        chunk: Dict[str, Any],
        chunk_index: int,
    ) -> ChunkScore:
        """
        Ask the LLM judge to score a single retrieved chunk.

        Args:
            query       : The original user query.
            chunk       : A dict from SearchService — keys: chunk_id, content, score, …
            chunk_index : Zero-based position in the ranked retrieval list.

        Returns:
            ChunkScore with LLM-assigned scores and reasoning.
        """
        content  = chunk.get("content", "")
        chunk_id = chunk.get("chunk_id", f"chunk_{chunk_index}")

        user_message = (
            f"Query: {query}\n\n"
            f"Retrieved chunk (rank {chunk_index + 1}):\n{content}"
        )

        logging.info(
            f"[Judge] Scoring chunk_id={chunk_id} "
            f"(rank={chunk_index + 1}, qdrant_score={chunk.get('score', 0):.3f})"
        )

        try:
            response = await self.llm.ainvoke(
                [
                    SystemMessage(content=CHUNK_JUDGE_SYSTEM),
                    HumanMessage(content=user_message),
                ],
                config={"max_tokens": 200},
            )

            raw = response.content.strip()
            logging.info(f"[Judge] Raw LLM response for chunk_id={chunk_id}: {raw!r}")

            parsed = json.loads(raw)

            return ChunkScore(
                chunk_id          = chunk_id,
                chunk_index       = chunk_index,
                content_preview   = content[:200],
                retrieval_score   = round(float(chunk.get("score", 0.0)), 4),
                context_relevance = round(float(parsed["context_relevance"]), 4),
                context_coverage  = round(float(parsed["context_coverage"]),  4),
                faithfulness      = round(float(parsed["faithfulness"]),       4),
                judge_reasoning   = parsed.get("reasoning", ""),
            )

        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logging.warning(
                f"[Judge] Failed to parse LLM response for chunk_id={chunk_id}: {e}. "
                f"Assigning neutral scores (0.5)."
            )
            return ChunkScore(
                chunk_id          = chunk_id,
                chunk_index       = chunk_index,
                content_preview   = content[:200],
                retrieval_score   = round(float(chunk.get("score", 0.0)), 4),
                context_relevance = 0.5,
                context_coverage  = 0.5,
                faithfulness      = 0.5,
                judge_reasoning   = f"Parse error — neutral score assigned. Raw: {raw!r}",
            )

        except Exception as e:
            logging.error(f"[Judge] Unexpected error scoring chunk_id={chunk_id}: {e}")
            raise

    async def score_all_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
    ) -> List[ChunkScore]:
        """
        Score all retrieved chunks for a query sequentially.

        Sequential (not concurrent) to respect API rate limits.
        Returns a list of ChunkScores in retrieval rank order.
        """
        if not chunks:
            logging.warning("[Judge] No chunks to score — returning empty list.")
            return []

        logging.info(f"[Judge] Scoring {len(chunks)} chunk(s) for query: '{query}'")

        scores = []
        for idx, chunk in enumerate(chunks):
            score = await self.score_chunk(query=query, chunk=chunk, chunk_index=idx)
            scores.append(score)

        logging.info(f"[Judge] Finished scoring {len(scores)} chunk(s).")
        return scores

    @staticmethod
    def aggregate(chunk_scores: List[ChunkScore]) -> Dict[str, Any]:
        """
        Compute aggregate metrics and verdict from a list of ChunkScores.

        Returns a dict with avg scores and a PASS / PARTIAL / FAIL verdict.
        """
        if not chunk_scores:
            return {
                "avg_context_relevance": 0.0,
                "avg_context_coverage": 0.0,
                "avg_faithfulness": 0.0,
                "overall_score": 0.0,
                "retrieval_verdict":"FAIL",
            }

        n = len(chunk_scores)
        avg_relevance  = round(sum(c.context_relevance for c in chunk_scores) / n, 4)
        avg_coverage   = round(sum(c.context_coverage  for c in chunk_scores) / n, 4)
        avg_faith      = round(sum(c.faithfulness       for c in chunk_scores) / n, 4)
        overall        = round((avg_relevance + avg_coverage + avg_faith) / 3,   4)

        if overall >= PASS_THRESHOLD:
            verdict = "PASS"
        elif overall >= PARTIAL_THRESHOLD:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

        logging.info(
            f"[Judge] Aggregate — relevance={avg_relevance}, "
            f"coverage={avg_coverage}, faithfulness={avg_faith}, "
            f"overall={overall}, verdict={verdict}"
        )

        return {
            "avg_context_relevance": avg_relevance,
            "avg_context_coverage":  avg_coverage,
            "avg_faithfulness":      avg_faith,
            "overall_score":         overall,
            "retrieval_verdict":     verdict,
        }