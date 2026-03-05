# ===================================================================================
# Project: VoiceRAG
# File: evals/schemas.py
# Description: Pydantic schemas for LLM-as-Judge evaluation system
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from pydantic import BaseModel, Field
from typing import List
from datetime import datetime


class EvalInput(BaseModel):
    """
    A single evaluation input
    """
    query: str = Field(..., description="The user query to evaluate.")
    user_id: str = Field(..., description="User ID for Qdrant tenant isolation.")
    session_id: str = Field(..., description="Session ID for Qdrant tenant isolation.")
    top_k: int = Field(default=5, description="Number of chunks to retrieve.")


class ChunkScore(BaseModel):
    """
    LLM judge scores for a single retrieved chunk.
    All scores are in [0.0, 1.0].
    """
    chunk_id: str = Field(..., description="Unique identifier for the chunk.")
    chunk_index: int = Field(..., description="Position index of the chunk in retrieval results.")
    content_preview: str = Field(..., description="First 200 chars of the chunk for readability.")
    retrieval_score: float = Field(..., description="Cosine similarity score from Qdrant.")

    # LLM judge scores
    context_relevance: float = Field(
        ..., ge=0.0, le=1.0,
        description="How relevant is this chunk to the query? (0=irrelevant, 1=perfectly relevant)"
    )
    context_coverage: float = Field(
        ..., ge=0.0, le=1.0,
        description="How much of the query does this chunk address? (0=nothing, 1=fully answers)"
    )
    faithfulness: float = Field(
        ..., ge=0.0, le=1.0,
        description="Is the chunk internally consistent and non-noisy? (0=noisy/corrupt, 1=clean)"
    )
    judge_reasoning: str = Field(..., description="One-line explanation from the LLM judge.")


class EvalResult(BaseModel):
    """
    Full evaluation result for a single query.
    """
    query: str
    user_id: str
    session_id: str
    total_chunks_retrieved: int

    chunk_scores: List[ChunkScore]

    # Aggregate scores across all retrieved chunks
    avg_context_relevance: float = Field(..., description="Mean context relevance across chunks.")
    avg_context_coverage: float = Field(..., description="Mean context coverage across chunks.")
    avg_faithfulness: float = Field(..., description="Mean faithfulness across chunks.")
    overall_score: float = Field(..., description="Mean of all three aggregate scores.")

    retrieval_verdict: str = Field(
        ..., description="PASS / PARTIAL / FAIL based on overall_score thresholds."
    )
    evaluated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class EvalReport(BaseModel):
    """
    Batch report aggregating multiple EvalResults.
    Written to JSON and rendered as Markdown.
    """
    total_queries: int
    passed: int
    partial: int
    failed: int

    avg_context_relevance: float
    avg_context_coverage: float
    avg_faithfulness: float
    avg_overall_score: float

    results: List[EvalResult]
    generated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())