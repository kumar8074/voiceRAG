# ===================================================================================
# Project: VoiceRAG
# File: src/services/indexing/doc_intel_indexer.py
# Description: Indexing pipeline using Sarvam Document Intelligence
# Author: LALAN KUMAR
# Created: [05-03-2026]
# Updated: [05-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import asyncio
from uuid import uuid4, uuid5, NAMESPACE_DNS
from typing import List, Dict, Any
from datetime import datetime, timezone
from time import time

from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from ..document_intelligence.sarvam_client import SarvamDocClient
from ..document_intelligence.md_chunker import MarkdownChunker
from ...logger import logging
from ...dependencies import get_qdrant_client, get_embedding_service
from ...config import QDRANT_COLLECTION_NAME


class DocIntelIndexer:
    """
    Indexing pipeline powered by Sarvam Document Intelligence.

    index_many:  fans out all uploaded PDFs to Sarvam concurrently,
                 collects markdown, chunks, embeds in batches, upserts to Qdrant.

    index_one:   single-file version (used internally by index_many).

    delete_document: mirrors QdrantIndexer.delete_document — same filter logic.
    """

    def __init__(self):
        self.doc_client = SarvamDocClient()
        self.chunker = MarkdownChunker()
        self.embedder = get_embedding_service()
        self.client = get_qdrant_client()

    async def index_many(
        self,
        files: List[Dict[str, str]],   # [{"file_path": ..., "original_name": ...}]
        user_id: str,
        session_id: str,
        language: str = "en-IN",
    ) -> List[Dict[str, Any]]:
        """
        Process up to MAX_FILES PDFs concurrently through Sarvam DI,
        then chunk → embed → upsert each one.

        Sarvam calls run in parallel (asyncio.gather).
        Chunking + embedding + upsert run per-file after Sarvam returns.

        Args:
            files:      List of dicts with 'file_path' and 'original_name'.
            user_id:    Multi-tenant isolation key.
            session_id: Session-level isolation key.
            language:   BCP-47 language code passed to Sarvam DI.

        Returns:
            List of per-file index result dicts.
        """
        if not files:
            raise ValueError("No files provided for indexing.")

        if not user_id:
            raise ValueError("user_id is required for multi-tenant indexing.")

        logging.info(
            f"[DocIntelIndexer] Starting concurrent Sarvam DI "
            f"for {len(files)} file(s), user={user_id}"
        )

        # Call Sarvam DI for all files concurrently 
        start_time = time()
        sarvam_tasks = [
            self.doc_client.process_pdf(f["file_path"])
            for f in files
        ]
        markdown_results: List[str] = await asyncio.gather(*sarvam_tasks)
        logging.info(
            f"[DocIntelIndexer] Sarvam DI completed for all {len(files)} "
            f"file(s) in {(time() - start_time):.2f}s"
        )

        # Chunk → Embed → Upsert per file 
        results = []
        for f, markdown in zip(files, markdown_results):
            result = await self._index_one(
                markdown=markdown,
                file_path=f["file_path"],
                original_name=f["original_name"],
                user_id=user_id,
                session_id=session_id,
            )
            results.append(result)

        logging.info(
            f"[DocIntelIndexer] All files indexed. "
            f"Total chunks: {sum(r['total_chunks'] for r in results)}"
        )
        return results

    def delete_document(self, user_id: str, doc_id: str) -> int:
        """
        Delete all Qdrant vectors for a specific document belonging to a user.
        Mirrors QdrantIndexer.delete_document — drop-in compatible.
        """
        delete_filter = Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="doc_id",  match=MatchValue(value=doc_id)),
            ]
        )

        logging.info(
            f"[DocIntelIndexer] Deleting doc_id={doc_id} for user={user_id}"
        )
        result = self.client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=delete_filter,
        )
        return result.status

    async def _index_one(
        self,
        markdown: str,
        file_path: str,
        original_name: str,
        user_id: str,
        session_id: str,
        doc_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Chunk → Embed → Upsert a single document's markdown into Qdrant.
        """
        if doc_id is None:
            doc_id = str(uuid4())

        logging.info(f"[DocIntelIndexer] Chunking: {original_name}")
        start_time = time()
        chunks = self.chunker.split(
            markdown=markdown,
            metadata={
                "user_id":   user_id,
                "session_id": session_id,
                "doc_id":    doc_id,
                "filename":  original_name,
            },
        )
        logging.info(
            f"[DocIntelIndexer] {len(chunks)} chunk(s) in "
            f"{(time() - start_time):.2f}s"
        )

        if not chunks:
            raise ValueError(f"Chunking produced no output for: {original_name}")

        # Embed
        logging.info(f"[DocIntelIndexer] Embedding {len(chunks)} chunk(s)...")
        start_time = time()
        embeddings = await self.embedder.embed_documents(chunks)
        logging.info(
            f"[DocIntelIndexer] Embeddings done in {(time() - start_time):.2f}s"
        )

        if len(embeddings) != len(chunks):
            raise RuntimeError(
                f"Embedding/chunk count mismatch for {original_name}: "
                f"{len(embeddings)} embeddings vs {len(chunks)} chunks."
            )

        # Build Qdrant points
        points: List[PointStruct] = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point_id = str(uuid5(NAMESPACE_DNS, f"{user_id}_{doc_id}_{idx}"))
            payload = {
                "user_id":     user_id,
                "session_id":  session_id,
                "doc_id":      doc_id,
                "chunk_id":    f"{doc_id}_{idx}",
                "chunk_index": idx,
                "content":     chunk.page_content,
                "filename":    original_name,
                "indexed_at":  datetime.now(timezone.utc).isoformat(),
            }
            points.append(PointStruct(id=point_id, vector=embedding, payload=payload))

        if not points:
            raise ValueError(f"No valid points to upsert for: {original_name}")

        # Upsert in batches of 100
        logging.info(f"[DocIntelIndexer] Upserting {len(points)} point(s) to Qdrant...")
        start_time = time()
        BATCH_SIZE = 100
        for i in range(0, len(points), BATCH_SIZE):
            self.client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=points[i : i + BATCH_SIZE],
            )
        logging.info(
            f"[DocIntelIndexer] Upsert complete in {(time() - start_time):.2f}s"
        )

        result = {
            "doc_id":         doc_id,
            "filename":       original_name,
            "total_chunks":   len(chunks),
            "indexed_chunks": len(points),
        }
        logging.info(f"[DocIntelIndexer] Indexed: {result}")
        return result
    
# Example usage:
if __name__ == "__main__":
    import asyncio
    indexer = DocIntelIndexer()
    results = asyncio.run(indexer.index_many(
        files=[{"file_path": "tmp/Startup India Kit_v5.pdf", "original_name": "Startup-kit"}],
        user_id="1",
        session_id="1"
    ))
    logging.info(f"Indexing results: {results}")