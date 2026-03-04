# ===================================================================================
# Project: VoiceRAG
# File: src/services/embedding/embedding_service.py
# Description: Gnerates embeddings for the documents
# Author: LALAN KUMAR
# Created: [03-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import numpy as np
from langchain_core.documents import Document
from typing import List
import asyncio

from ...config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSION
from ...logger import logging

def normalize(vectors: List[List[float]]) -> List[List[float]]:
    """Convert embeddings to unit vectors."""
    arr = np.array(vectors)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / norms).tolist()

class EmbeddingService:
    def __init__(self, model=EMBEDDING_MODEL, batch_size=EMBEDDING_BATCH_SIZE, embedding_dim=EMBEDDING_DIMENSION):
        self.model=model
        self.batch_size=batch_size
        self.embedding_dim=embedding_dim
        self.semaphore = asyncio.Semaphore(5)  # max concurrent requests
    
    # Batch processor
    async def _embed_batch(self,texts:list[str])->List[list[float]]:
        async with self.semaphore:
            embeddings=await self.model.feature_extraction(texts)
        
        if isinstance(embeddings, np.ndarray):
            embeddings=embeddings.tolist()
         
        # For query   
        if isinstance(embeddings[0], float):
            embeddings = [embeddings]
        
        if len(embeddings[0])!=self.embedding_dim:
            logging.error(f"Expected embedding dimension {self.embedding_dim}, got {len(embeddings[0])}")
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {len(embeddings[0])}")
        
        # Normalize
        normalized_embeddings=normalize(embeddings)
        return normalized_embeddings        
        

    # Embed documents in batches
    async def embed_documents(self, documents:List[Document])->List[list[float]]:
        if not documents:
            return []
        
        all_embeddings=[]
        tasks=[]
        total=len(documents)
        
        for i in range(0, total, self.batch_size):
            batch_docs=documents[i:i+self.batch_size]
            batch_texts=[f"passage: {doc.page_content}" for doc in batch_docs]
            
            logging.info(f"Scheduling batch {i // self.batch_size + 1} ({len(batch_docs)} documents)")
            
            tasks.append(self._embed_batch(batch_texts))
            
        # Run all embedding requests concurrently
        results = await asyncio.gather(*tasks)
            #batch_embeddings=await self._embed_batch(batch_texts)
            #all_embeddings.extend(batch_embeddings)
        
        # Flatten results
        for batch in results:
            all_embeddings.extend(batch)
        
        return all_embeddings
    
    # Embed query text (for later similarity search)
    async def embed_query(self, query:str)->list[float]:
        if not query:
            raise ValueError("Query text cannot be empty")
        
        # Required prefix for E5 models
        prefixed_query = f"query: {query}"
        
        query_embedding= (await self._embed_batch([prefixed_query]))[0] # Ensure same dimension as document embeddings
        
        return query_embedding
    
# Example usage:
async def main():
    from ..pdf_parser.parser import PDFParser
    from ..chunker.text_chunker import TextChunker
    from time import time
    
    start_time = time()
    docs=await PDFParser.parse_pdf("tmp/83577772-b923-4c40-891d-2edd3fe26fad_Startup India Kit_v5.pdf")
    end_time = time()
    logging.info(f"PDF parsing took {(end_time - start_time):.2f} seconds for {len(docs)} documents.")
    
    start_time = time()
    chunker=TextChunker()
    chunks=chunker.split(docs)
    end_time = time()
    logging.info(f"Text chunking took {(end_time - start_time):.2f} seconds for {len(chunks)} chunks.")

    start_time = time()
    embedding_service=EmbeddingService()
    embeddings=await embedding_service.embed_documents(chunks)
    end_time = time()
    logging.info(f"Embedding generation took {(end_time - start_time):.2f} seconds for {len(chunks)} chunks.")
    
    logging.info(f"Generated {len(embeddings)} embeddings for {len(chunks)} chunks.")
    logging.info(f"Embedding dimension: {len(embeddings[0])}")
    
    start_time = time()
    query_embedding=await embedding_service.embed_query("What are the benefits of Startup India initiative?")
    end_time = time()
    logging.info(f"Generated query embedding of dimension: {len(query_embedding)}")
    logging.info(f"Query embedding generation took {(end_time - start_time):.2f} seconds.")
    
    


if __name__ == "__main__":
    asyncio.run(main())
    
