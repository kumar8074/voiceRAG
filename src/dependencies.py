# ===================================================================================
# Project: VoiceRAG
# File: src/config.py
# Description: Shared application dependencies (connect once, reuse everywhere)
# Author: LALAN KUMAR
# Created: [04-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from functools import lru_cache

from .services.embedding.embedding_service import EmbeddingService
from .services.qdrant.factory import QdrantFactory
from .config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_GRPC_PORT,
    QDRANT_PREFER_GRPC,
    QDRANT_TIMEOUT
)


@lru_cache(maxsize=1)
def get_qdrant_client():
    """
    Create a single reusable Qdrant client.
    """
    client, _ = QdrantFactory.connect(
        host=QDRANT_HOST,
        port=QDRANT_PORT,
        grpc_port=QDRANT_GRPC_PORT,
        prefer_grpc=QDRANT_PREFER_GRPC,
        timeout=QDRANT_TIMEOUT
    )

    # Ensure collection exists once
    QdrantFactory.ensure_collection(client)

    return client


@lru_cache(maxsize=1)
def get_embedding_service():
    """
    Shared embedding service instance.
    """
    return EmbeddingService()