# ===================================================================================
# Project: VoiceRAG
# File: src/services/qdrant/collection_config.py
# Description: Qdrant collection configuration
# Author: LALAN KUMAR
# Created: [03-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from qdrant_client.models import Distance, VectorParams
from ...config import EMBEDDING_DIMENSION

def get_collection_config() -> VectorParams:
    """
    Returns vector configuration for Qdrant collection.
    """

    return VectorParams(
        size=EMBEDDING_DIMENSION,
        distance=Distance.COSINE  # E5 embeddings are normalized
    )