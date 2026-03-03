# ===================================================================================
# Project: VoiceRAG
# File: src/services/qdrant/factory.py
# Description: Connects to Qdrant
# Author: LALAN KUMAR
# Created: [03-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

from typing import Tuple, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from ...config import (
    QDRANT_HOST,
    QDRANT_PORT,
    QDRANT_COLLECTION_NAME,
    QDRANT_GRPC_PORT,
    QDRANT_PREFER_GRPC,
    QDRANT_TIMEOUT
)
from .collection_config import get_collection_config

class QdrantFactory:
    """
    Factory class to create and validate Qdrant connection.
    """
    @staticmethod
    def connect(
        host: str = QDRANT_HOST,
        port: int = QDRANT_PORT,
        grpc_port: int = QDRANT_GRPC_PORT,
        prefer_grpc: bool = QDRANT_PREFER_GRPC,
        timeout: int = QDRANT_TIMEOUT
    ) -> Tuple[QdrantClient, Dict[str, Any]]:
        """
        Establish connection to Qdrant and validate service availability.

        Returns:
            Tuple[QdrantClient, health_info]
        """

        print("Connecting to Qdrant...")

        try:
            client = QdrantClient(
                host=host,
                port=port,
                grpc_port=grpc_port,
                prefer_grpc=prefer_grpc,
                timeout=timeout
            )

            # Health check
            health = client.get_collections()

            print("Qdrant connection established successfully.")
            return client, {"status": "healthy", "collections": health.collections}

        except UnexpectedResponse as e:
            print(f"Qdrant returned unexpected response: {e}")
            raise

        except Exception as e:
            print(f"Failed to connect to Qdrant at {host}:{port}: {e}")
            raise


    @staticmethod
    def ensure_collection(client: QdrantClient) -> None:
        """
        Create collection if it does not exist.
        """

        existing_collections = [c.name for c in client.get_collections().collections]

        if QDRANT_COLLECTION_NAME in existing_collections:
            print(f"Collection already exists: {QDRANT_COLLECTION_NAME}")
            return

        print(f"Creating collection: {QDRANT_COLLECTION_NAME}")

        client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=get_collection_config()
        )

        print(f"Collection created successfully: {QDRANT_COLLECTION_NAME}")
        
        
# Example usage:
if __name__ == "__main__":
    client, _ = QdrantFactory.connect()
    QdrantFactory.ensure_collection(client)