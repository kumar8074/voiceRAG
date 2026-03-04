# ===================================================================================
# Project: VoiceRAG
# File: src/services/search/search_service.py
# Description: Vector search service using Qdrant
# Author: LALAN KUMAR
# Created: [04-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from typing import List, Dict, Any
from time import time

from qdrant_client.models import Filter, FieldCondition, MatchValue

from ...config import QDRANT_COLLECTION_NAME
from ...logger import logging
from ...dependencies import get_qdrant_client, get_embedding_service


class SearchService:
    """
    Pure vector search service for VoiceRAG.
    Performs:
        - Query embedding
        - Qdrant ANN search
        - User-level isolation
    """

    def __init__(self):
        self.embedder = get_embedding_service()

        self.client = get_qdrant_client()

    async def search(
        self,
        query: str,
        user_id: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Perform vector similarity search restricted to a specific user.

        Returns:
            List of relevant chunks with metadata.
        """

        if not query:
            logging.error("No Search Query")
            raise ValueError("Query cannot be empty")

        if not user_id:
            logging.error("user_id is required for multi-tenant isolation")
            raise ValueError("user_id is required for multi-tenant isolation")

        # Embed query
        logging.info("Initiating query embedding...")
        start_time=time()
        query_vector = await self.embedder.embed_query(query)
        end_time=time()
        logging.info(f"Query Embedding sucessful in {(end_time-start_time):.2f}seconds")

        # Build user filter
        logging.info("Building user filter...")
        start_time=time()
        user_filter = Filter(
            must=[
                FieldCondition(
                    key="user_id",
                    match=MatchValue(value=user_id)
                )
            ]
        )

        # Search Qdrant
        logging.info("Initiating search...")
        results = self.client.query_points(
            collection_name=QDRANT_COLLECTION_NAME,
            query=query_vector,
            limit=top_k,
            query_filter=user_filter
        )

        # Format results
        formatted_results = []
        logging.info("Formatting Result...")
        for hit in results.points:
            formatted_results.append({
                "score": hit.score,
                "chunk_id": hit.payload.get("chunk_id"),
                "doc_id": hit.payload.get("doc_id"),
                "content": hit.payload.get("content"),
                "language": hit.payload.get("language"),
                "chunk_index": hit.payload.get("chunk_index"),
            })

        end_time=time()
        logging.info(f"Search completed in {(end_time-start_time):.2f}seconds")
        logging.info(f"Search Results:{formatted_results}")
        return formatted_results
    
# Example usage:
if __name__ == "__main__":
    import asyncio
    searcher=SearchService()
    
    asyncio.run(searcher.search(query="Startup India initiative ke kya fayde hai?", user_id="1"))