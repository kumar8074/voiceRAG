# ===================================================================================
# Project: VoiceRAG
# File: src/services/embedding/factory.py
# Description: Generates embeddings for the documents using SentenceTransformer (Used for testing latency)
# This would be better when self-hosted on on-premise GPU and scaled horizontally 
# but as of now HF inference provider API is faster
# Author: LALAN KUMAR
# Created: [03-03-2026]
# Updated: [03-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import numpy as np
from typing import List
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from ...config import EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSION


class EmbeddingService:
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        batch_size: int = EMBEDDING_BATCH_SIZE,
        embedding_dim: int = EMBEDDING_DIMENSION,
        normalize: bool = True,
    ):
        self.model = SentenceTransformer(model_name)
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.normalize = normalize

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

        if isinstance(embeddings, np.ndarray):
            if embeddings.ndim == 1:
                embeddings = np.expand_dims(embeddings, axis=0)

            if embeddings.shape[1] != self.embedding_dim:
                raise ValueError(
                    f"Expected embedding dimension {self.embedding_dim}, "
                    f"got {embeddings.shape[1]}"
                )

            return embeddings.tolist()

        raise ValueError(f"Unexpected embedding type: {type(embeddings)}")


    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        if not documents:
            return []

        all_embeddings = []
        total = len(documents)

        for i in range(0, total, self.batch_size):
            batch_docs = documents[i : i + self.batch_size]
            batch_texts = [
                f"passage: {doc.page_content}" for doc in batch_docs
            ]

            for idx in range(len(batch_docs)):
                global_index = i + idx + 1
                print(f"Embedding document {global_index}/{total}")

            batch_embeddings = self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return all_embeddings


    def embed_query(self, query: str) -> List[float]:
        if not query:
            raise ValueError("Query text cannot be empty")

        prefixed_query = f"query: {query}"
        query_embedding = self._embed_batch([prefixed_query])[0]

        return query_embedding



# Example usage:
if __name__ == "__main__":
    from ..pdf_parser.parser import PDFParser
    from ..chunker.text_chunker import TextChunker
    from time import time

    start_time = time()
    docs = PDFParser.parse_pdf(
        "tmp/83577772-b923-4c40-891d-2edd3fe26fad_Startup India Kit_v5.pdf"
    )
    end_time = time()
    print(f"PDF parsing took {end_time - start_time:.2f} seconds for {len(docs)} documents.")

    start_time = time()
    chunker = TextChunker()
    chunks = chunker.split(docs)
    end_time = time()
    print(f"Text chunking took {end_time - start_time:.2f} seconds for {len(chunks)} chunks.")

    start_time = time()
    embedding_service = EmbeddingService()
    embeddings = embedding_service.embed_documents(chunks)
    end_time = time()
    print(f"Embedding generation took {end_time - start_time:.2f} seconds for {len(chunks)} chunks.")

    print(f"Generated {len(embeddings)} embeddings for {len(chunks)} chunks.")
    print(f"Embedding dimension: {len(embeddings[0])}")

    start_time = time()
    query_embedding = embedding_service.embed_query(
        "What are the benefits of Startup India initiative?"
    )
    end_time = time()
    print(f"Generated query embedding of dimension: {len(query_embedding)}")
    print(f"Query embedding generation took {end_time - start_time:.2f} seconds.")