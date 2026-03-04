# ===================================================================================
# Project: VoiceRAG
# File: src/services/indexing/indexer.py
# Description: Indexing pipeline (Parse → Chunk → Embed → Qdrant)
# Author: LALAN KUMAR
# Created: [04-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from uuid import uuid4, uuid5, NAMESPACE_DNS
from typing import List, Dict, Any
from datetime import datetime, timezone
from time import time

from qdrant_client.models import PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue

from ..pdf_parser.parser import PDFParser
from ..chunker.text_chunker import TextChunker

from ...logger import logging
from ...dependencies import get_qdrant_client, get_embedding_service
from ...config import QDRANT_COLLECTION_NAME

class QdrantIndexer:
    """
    - Parses PDF
    - Chunks text
    - Generates embeddings
    - Upserts into Qdrant
    """

    def __init__(self):
        self.parser = PDFParser()
        self.chunker = TextChunker()
        self.embedder = get_embedding_service()

        self.client = get_qdrant_client()

    async def index_pdf(
        self,
        file_path: str,
        user_id: str,
        doc_id: str | None = None,
        language: str = "en"
    ) -> Dict[str, Any]:

        if not user_id:
            logging.error("user_id is required for multi-tenant indexing")
            raise ValueError("user_id is required for multi-tenant indexing")

        if doc_id is None:
            doc_id = str(uuid4())

        # Parse
        logging.info("Initiating PDF Parsing...")
        start_time=time()
        documents = await self.parser.parse_pdf(file_path)
        end_time=time()
        logging.info(f"Parsing Completed in {(end_time-start_time):2f}seconds")
        if not documents:
            logging.error("No content extracted from PDF.")
            raise ValueError("No content extracted from PDF.")

        # Chunk
        logging.info("Initiating Documents chunking...")
        start_time=time()
        chunks = self.chunker.split(documents)
        end_time=time()
        logging.info(f"Documents chunking completed in {(end_time-start_time):.2f}seconds")
        if not chunks:
            logging.error("Chunking produced no output.")
            raise ValueError("Chunking produced no output.")

        # Embed
        logging.info("Initiating chunks embedding process...")
        start_time=time()
        embeddings = await self.embedder.embed_documents(chunks)
        end_time=time()
        logging.info(f"Embeddings generated in {(end_time-start_time):.2f}seconds")

        if len(embeddings) != len(chunks):
            logging.error("Mismatch between chunks and embeddings count.")
            raise RuntimeError("Mismatch between chunks and embeddings count.")

        logging.info("Preparing Qdrant for Indexing...")
        start_time=time()
        # Prepare Qdrant points
        points: List[PointStruct] = []

        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):

            point_id = str(uuid5(NAMESPACE_DNS, f"{user_id}_{doc_id}_{idx}"))

            payload = {
                "user_id": user_id,
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{idx}",
                "chunk_index": idx,
                "content": chunk.page_content,
                "language": language,
                "indexed_at": datetime.now(timezone.utc).isoformat()
            }

            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                )
            )

        if not points:
            logging.error("No valid chunks to index.")
            raise ValueError("No valid chunks to index.")

        # Upsert in batches
        BATCH_SIZE = 100

        for i in range(0, len(points), BATCH_SIZE):
            batch = points[i:i + BATCH_SIZE]

            self.client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=batch
            )

        result = {
            "user_id": user_id,
            "doc_id": doc_id,
            "total_chunks": len(chunks),
            "indexed_chunks": len(points)
        }
        end_time=time()
        logging.info(f"Indexing completed in {(end_time-start_time):.2f}seconds")
        logging.info(f"Indexing result: {result}")

        return result

    def delete_document(self, user_id: str, doc_id: str) -> int:
        """
        Delete all chunks belonging to a specific document for a user.
        """

        delete_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                ),
                FieldCondition(
                    key="doc_id",
                    match=MatchValue(value=doc_id)
                )
            ]
        )

        logging.info(f"Initiating index deletion for {user_id}...")
        result = self.client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=delete_filter
        )

        return result.status

# Example usage:
if __name__=="__main__":
    import asyncio
    indexer=QdrantIndexer()
    result=asyncio.run(indexer.index_pdf(file_path="tmp/83577772-b923-4c40-891d-2edd3fe26fad_Startup India Kit_v5.pdf", user_id="1"))
    
    