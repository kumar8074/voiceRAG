# VoiceRAG

> A multilingual, voice-enabled Retrieval-Augmented Generation system for conversational document Q&A — supporting Hindi, English, and Indian regional languages.

Built with FastAPI, LangGraph, Qdrant, PostgreSQL, and the Sarvam AI stack.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup Guide](#setup-guide)
  - [Prerequisites](#prerequisites)
  - [Environment Variables](#environment-variables)
  - [Running with Docker Compose](#running-with-docker-compose)
- [API Reference](#api-reference)
- [Evaluation System](#evaluation-system)
  - [How It Works](#how-it-works)
  - [Running Evals Manually](#running-evals-manually)
  - [Production Background Evals](#production-background-evals)
  - [Eval Reports](#eval-reports)
- [Logging](#logging)
- [Observed Performance During Development](#observed-performance)

---

## Overview

VoiceRAG lets users upload PDF documents and query them via text or voice in multiple Indian languages. The system automatically detects the query language, retrieves semantically relevant document chunks, generates a grounded response, and speaks it back via text-to-speech — all in real time.

Key capabilities:

- **Multilingual** — Hindi, English, Tamil, Telugu, Bengali, Marathi, Kannada (and their code-mixed variants)
- **Voice I/O** — streaming ASR via Sarvam Saaras v3, TTS via Sarvam, barge-in support
- **Agentic RAG** — LangGraph-based pipeline with intent classification, relevance grading, and automatic query rewriting + retry
- **Document Intelligence** — Sarvam Document Intelligence for high-fidelity PDF-to-Markdown conversion
- **Multi-tenant** — all Qdrant vectors and conversation history are isolated by `user_id` + `session_id`
- **Production Evals** — LLM-as-Judge retrieval evaluation runs silently in the background after every document query

---

## Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser UI                           │
│              static/index.html  (DM Sans, dark)             │
│   Text Chat ──────────────────── Voice Modal (WebSocket)    │
└────────────┬──────────────────────────────┬─────────────────┘
             │  POST /api/v1/chat           │  WS /ws/voice
             │  (SSE token stream)          │  (binary PCM in, JSON + MP3 out)
             ▼                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI  (src/main.py)                 │
│                                                             │
│   /api/v1/upload        →  file_upload router               │
│   /api/v1/chat          →  chat router                      │
│   /api/v1/index-status  →  job polling                      │
│   /api/v1/document      →  delete document                  │
│   /ws/voice             →  voice WebSocket router           │
└───────────┬─────────────────────────────────────────────────┘
            │
  ┌─────────┴──────────┐
  │                    │
  ▼                    ▼
RAGService          Voice Pipeline
(agentic_rag.py)    (pipeline.py)
  │                    │
  ├── SearchService    ├── ASRSession (Sarvam Saaras v3)
  ├── ConvStore        ├── RAGService
  └── LangGraph        └── TTS (Sarvam Streaming API)
         │
    ┌────┴─────────────────────────────┐
    │      LangGraph State Machine     │
    │                                  │
    │  classify_intent                 │
    │     ├── general_response         │
    │     └── fetch_history            │
    │            └── search            │
    │                 └── verify_relevance
    │                      ├── generate│
    │                      ├── rewrite_query → search (retry)
    │                      └── handle_no_results
    └──────────────────────────────────┘
             │
             ▼
      _save_turn()
             │
             └──→ fire_eval()  [background, zero latency]
                       │
                  EvalRunner → RetrievalJudge → Report

┌───────────────┐   ┌────────────────┐   ┌──────────────────┐
│   PostgreSQL  │   │     Qdrant     │   │  Sarvam AI APIs  │
│  (conv hist)  │   │  (vectors)     │   │  LLM / ASR / TTS │
│  SQLModel     │   │  Cosine / 1024d│   │  / Doc Intel     │
└───────────────┘   └────────────────┘   └──────────────────┘
```
---

## Project Structure

```
VoiceRAG/
│
├── src/
│   ├── main.py                          # FastAPI app, lifespan, router registration
│   ├── config.py                        # Pydantic Settings, model init, env exports
│   ├── dependencies.py                  # LRU-cached Qdrant + Embedding singletons
│   ├── logger.py                        # Timestamped file logger
│   │
│   ├── db/
│   │   ├── factory.py                   # Async SQLAlchemy engine + session factory
│   │   └── init_db.py                   # SQLModel metadata.create_all on startup
│   │
│   ├── models/
│   │   └── conversation.py              # ConversationMessage SQLModel table
│   │
│   ├── schemas/api/
│   │   └── request_response.py          # ChatRequest, ASRRequest, TTSRequest Pydantic models
│   │
│   ├── routers/
│   │   ├── file_upload.py               # POST /upload, GET /index-status, DELETE /document
│   │   ├── chat.py                      # POST /chat  (SSE streaming)
│   │   └── voice.py                     # WS /ws/voice  (full duplex voice pipeline)
│   │
│   └── services/
│       ├── document_intelligence/
│       │   ├── sarvam_client.py         # Async Sarvam DI wrapper (create→upload→poll→download)
│       │   └── md_chunker.py            # Heading-aware Markdown chunker + junk filter
│       │
│       ├── embedding/
│       │   └── embedding_service.py     # HF Inference E5-large, batched, normalized
│       │
│       ├── qdrant/
│       │   ├── factory.py               # QdrantFactory.connect() + ensure_collection()
│       │   └── collection_config.py     # VectorParams: 1024d, Cosine distance
│       │
│       ├── indexing/
│       │   └── doc_intel_indexer.py     # Orchestrates: DI → chunk → embed → upsert
│       │
│       ├── search/
│       │   └── search_service.py        # embed query → Qdrant ANN with user/session filter
│       │
│       ├── rag/
│       │   ├── states.py                # RAGState TypedDict for LangGraph
│       │   ├── prompts.py               # All system prompts (intent, grader, decomposer, RAG)
│       │   ├── prompt_builder.py        # Builds final RAG prompt from context + history
│       │   ├── conversation_store.py    # Postgres read/write for conversation history
│       │   ├── rag_service.py           # Simple RAG (superseded, kept for reference)
│       │   └── agentic_rag.py           # LangGraph agentic RAG — active implementation
│       │
│       └── voice/
│           ├── asr_handler.py           # Sarvam Saaras v3 streaming ASR session
│           ├── tts_handler.py           # Sarvam TTS, MP3 streaming
│           ├── audio_utils.py           # PCM→base64, Unicode language detection
│           └── pipeline.py              # Orchestrates ASR → RAG → TTS per voice turn
│
├── evals/
│   ├── __init__.py
│   ├── schemas.py                       # EvalInput, ChunkScore, EvalResult, EvalReport
│   ├── judges.py                        # RetrievalJudge: LLM scores each chunk (3 dimensions)
│   ├── eval_runner.py                   # EvalRunner: search → judge → JSON + MD reports
│   ├── background_eval.py               # fire_eval(): fire-and-forget production trigger
│   ├── run_eval.py                      # CLI entry point
│   └── queries.txt                      # Sample eval queries (one per line)
│
├── evals/reports/                       # Auto-created; eval JSON + MD written here
│
├── static/
│   └── index.html                       # Single-file UI (sidebar, chat, voice modal)
│
├── logs/                                # Auto-created; timestamped .log files
├── tmp/                                 # Auto-created; uploaded PDF staging area
│
├── Dockerfile                           # Multi-stage: uv builder → python:3.12-slim runtime
├── docker-compose.yml                   # VoiceRAG app + Qdrant + PostgreSQL
└── .env                                 # All secrets and config (see below)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| LLM | Sarvam `sarvam-m` via OpenAI-compatible API |
| ASR | Sarvam Saaras v3 (streaming WebSocket) |
| TTS | Sarvam TTS |
| Document Intelligence | Sarvam Document Intelligence |
| Embeddings | `intfloat/multilingual-e5-large` (1024d) via HF Inference |
| Vector DB | Qdrant (Cosine, REST) |
| Relational DB | PostgreSQL 16 + asyncpg + SQLModel |
| RAG orchestration | LangGraph |
| LLM client | LangChain / langchain-openai |
| Eval | Custom LLM-as-Judge (judges.py + eval_runner.py) |
| Containerisation | Docker + Docker Compose |
| Python packaging | uv |

---

## Setup Guide

### Prerequisites

- Docker Desktop (or Docker Engine + Docker Compose v2)
- A Sarvam AI API key — [https://dashboard.sarvam.ai](https://dashboard.sarvam.ai)
- A Hugging Face token with Inference API access — [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

No local Python installation is required for running the app — everything runs inside Docker.

### Environment Variables

Create a `.env` file in the project root:

```env
# Sarvam AI 
SARVAM_API_KEY=your_sarvam_api_key_here

# Hugging Face (E5 embeddings via Inference API)
HF_TOKEN=your_hf_token_here

# PostgreSQL 
POSTGRES_DB=voicerag_db
POSTGRES_USER=voicerag_user
POSTGRES_PASSWORD=voicerag

# Qdrant (overridden by docker-compose to use service name)
QDRANT_HOST=localhost
QDRANT_PORT=6333

# LangSmith (optional, for tracing) 
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=voicerag

# Optional overrides (defaults shown) 
# CHUNKER_CHUNK_SIZE=800
# CHUNKER_CHUNK_OVERLAP=100
# EMBEDDING_BATCH_SIZE=32
# LLM_MAX_TOKENS=2048
```

> The `docker-compose.yml` overrides `QDRANT_HOST=qdrant` and `POSTGRES_HOST=postgres` automatically so internal service names are used inside Docker. Your `.env` values for these are only used for local (non-Docker) runs.

### Running with Docker Compose

**1. Clone the repository and create your `.env` file** (see above).

**2. Build and start all services:**

```bash
docker compose up --build
```

This starts three containers: `voicerag-qdrant`, `voicerag-postgres`, and `voicerag-app`. The app container waits for both databases to pass their healthchecks before starting.

**3. Open the UI:**

```
http://localhost:8000
```

**4. Verify services are healthy:**

```bash
docker compose ps
```

All three services should show `healthy`.

**5. View live logs:**

```bash
# All services
docker compose logs -f

# App only
docker compose logs -f voicerag
```

**6. Stop all services:**

```bash
docker compose down
```

**7. Stop and remove all data volumes:**

```bash
docker compose down -v
```

> The app runs a single Uvicorn worker. This is intentional — the in-memory `_jobs` dict in `file_upload.py` is not safe for multi-worker deployments. To scale to multiple workers, move job state to Redis or PostgreSQL and then increase `--workers`.

---

## API Reference

### File Upload

```
POST /api/v1/upload
Content-Type: multipart/form-data
```

```
GET /api/v1/index-status/{job_id}
```

```
DELETE /api/v1/document/{doc_id}?user_id=...
```

### Chat

```
POST /api/v1/chat
Content-Type: application/json
```

### Voice

```
WS /ws/voice?user_id=...&session_id=...

Browser → Server:
  Binary frames : raw PCM 16kHz s16le audio chunks
  Text frame    : { "type": "stop" }

Server → Browser:
  { "type": "ready" }
  { "type": "listening" }
  { "type": "speech_end" }
  { "type": "transcript", "text": "...", "language_code": "hi-IN"|"en-IN" }
  { "type": "rag_start" }
  { "type": "token", "text": "..." }
  { "type": "rag_done" }
  { "type": "tts_start", "language_code": "..." }
  { "type": "tts_done" }
  { "type": "barge_in" }
  { "type": "error", "message": "..." }
  Binary frames : MP3 audio chunks (TTS output)
```

---

## Evaluation System

### How It Works

VoiceRAG ships with a zero-dataset LLM-as-Judge evaluation system that measures retrieval quality automatically — no ground truth labels needed.

The judge evaluates each retrieved chunk on three dimensions:

| Dimension | What it measures | Range |
|---|---|---|
| **Context Relevance** | Is this chunk topically relevant to the query? | 0.0 – 1.0 |
| **Context Coverage** | How much of the query does this chunk address? | 0.0 – 1.0 |
| **Faithfulness** | Is the chunk internally coherent and non-noisy? | 0.0 – 1.0 |

The three per-chunk scores are averaged to give a query-level `overall_score`. Verdicts are assigned as:

| Verdict | Threshold |
|---|---|
| ✅ PASS | overall_score ≥ 0.7 |
| ⚠️ PARTIAL | 0.4 ≤ overall_score < 0.7 |
| ❌ FAIL | overall_score < 0.4 |

### Running Evals Manually

Evals are run from the project root against a live Qdrant instance (documents must already be indexed).

**Batch from file (one query per line):**

```bash
python -m evals.run_eval \
    --queries-file evals/queries.txt \
    --user-id 1 \
    --session-id 1 \
    --top-k 5
```

A ready-made `evals/queries.txt` with 30 queries derived from the Startup India Kit document is included.

### Production Background Evals

In production, evals fire automatically after every document query — with zero impact on user-perceived latency.

The trigger is in `agentic_rag.py`. After `_save_turn()` commits the conversation turn to Postgres, `fire_eval()` is called for document queries only:


`fire_eval()` schedules an `asyncio.Task` on the running event loop and returns immediately. The eval coroutine runs in the background, and any failure is caught and logged as a `WARNING` — it can never crash the application.

The `EvalRunner` singleton is initialised once (lazy, on first eval) and reused across all background tasks, so there is no reconstruction overhead per query.

### Eval Reports

Every eval run produces two files in `evals/reports/`:

**JSON** (`eval_<timestamp>.json`) — full machine-readable report with all per-chunk scores, reasoning, and aggregates. Suitable for dashboards or further analysis.

**Markdown** (`eval_<timestamp>.md`) — human-readable summary with a batch table up top followed by a per-query drill-down showing each chunk's three scores, its Qdrant similarity score, and the judge's one-line reasoning.

Example Markdown summary:

```
# VoiceRAG — Retrieval Eval Report

**Generated at:** 2026-03-05T12:15:30
**Total queries:** 30

## Batch Summary

| Metric           | Score  |
|------------------|--------|
| ✅ Passed        | 24 / 30 |
| ⚠️ Partial       | 4 / 30  |
| ❌ Failed        | 2 / 30  |
| Avg Relevance    | 0.8210 |
| Avg Coverage     | 0.7640 |
| Avg Faithfulness | 0.9100 |
| **Avg Overall**  | **0.8317** |
```

All eval activity is also written to the application logs in `logs/` with the `[BackgroundEval]` and `[EvalRunner]` prefixes for easy filtering.

---

## Logging

Logs are written to `logs/<timestamp>.log` (auto-created). Every module uses the same logger from `src/logger.py` 

```bash
# Example: tail only RAG graph events from Docker
docker compose logs -f voicerag | grep "\[Graph\]"

# Tail eval activity
docker compose logs -f voicerag | grep "\[BackgroundEval\]\|\[EvalRunner\]\|\[Judge\]"
```

---

## Observed-Performance

- The Response time for `Simple Queries` was <1s, thanks to real-time `streaming`
- With FastAPI's `async` capabilities it can support `N` user's and is highly scalable, Not only limited to 5 concurrent users
- User's are allowed to upload max of `3 files each of 50MB max` per session
- Major latency contributor is `Embedding Serice` as it's relied on `HF_INFERENCE_PROVIDER` which goes through network and have latency so `24-28s (see attached logs)`, This can be significantly brought down by self-hosting the `Embedding Model` on Nvidia GPU
- The retrieval performance also seems to be Good, and latency depends on the size of content present in the files. I have tested it on research paper latency seem to be `3-5s` but thanks to streaming `User Perceieved latency seems negligible`
