# ===================================================================================
# Project: VoiceRAG
# File: src/services/embedding/embedding_service.py
# Description: Gnerates embeddings for the documents
# Author: LALAN KUMAR
# Created: [03-03-2026]
# Updated: [03-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import numpy as np
from langchain_core.documents import Document
from typing import List
from ...config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, EMBEDDING_DIMENSION

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
    
    # Batch processor
    def _embed_batch(self,texts:list[str])->List[list[float]]:
        embeddings=self.model.feature_extraction(texts)
        
        if isinstance(embeddings, np.ndarray):
            embeddings=embeddings.tolist()
         
        # For query   
        if isinstance(embeddings[0], float):
            embeddings = [embeddings]
        
        if len(embeddings[0])!=self.embedding_dim:
            raise ValueError(f"Expected embedding dimension {self.embedding_dim}, got {len(embeddings[0])}")
        
        # Normalize
        normalized_embeddings=normalize(embeddings)
        return normalized_embeddings        
        

    # Embed documents in batches
    def embed_documents(self, documents:List[Document])->List[list[float]]:
        if not documents:
            return []
        
        all_embeddings=[]
        total=len(documents)
        
        for i in range(0, total, self.batch_size):
            batch_docs=documents[i:i+self.batch_size]
            batch_texts=[f"passage: {doc.page_content}" for doc in batch_docs]
            
            print(f"Processing batch {i // self.batch_size + 1} ({len(batch_docs)} documents)")
                
            batch_embeddings=self._embed_batch(batch_texts)
            all_embeddings.extend(batch_embeddings)
        
        return all_embeddings
    
    # Embed query text (for later similarity search)
    def embed_query(self, query:str)->list[float]:
        if not query:
            raise ValueError("Query text cannot be empty")
        
        # Required prefix for E5 models
        prefixed_query = f"query: {query}"
        
        query_embedding=self._embed_batch([prefixed_query])[0] # Ensure same dimension as document embeddings
        
        return query_embedding
    
    
    
# Example usage:
if __name__ == "__main__":
    from ..pdf_parser.parser import PDFParser
    from ..chunker.text_chunker import TextChunker
    from time import time
    
    start_time = time()
    docs=PDFParser.parse_pdf("tmp/83577772-b923-4c40-891d-2edd3fe26fad_Startup India Kit_v5.pdf")
    end_time = time()
    print(f"PDF parsing took {end_time - start_time:.2f} seconds for {len(docs)} documents.")
    
    start_time = time()
    chunker=TextChunker()
    chunks=chunker.split(docs)
    end_time = time()
    print(f"Text chunking took {end_time - start_time:.2f} seconds for {len(chunks)} chunks.")
    
    start_time = time()
    embedding_service=EmbeddingService()
    embeddings=embedding_service.embed_documents(chunks)
    end_time = time()
    print(f"Embedding generation took {end_time - start_time:.2f} seconds for {len(chunks)} chunks.")
    
    print(f"Generated {len(embeddings)} embeddings for {len(chunks)} chunks.")
    print(f"Embedding dimension: {len(embeddings[0])}")
    
    start_time = time()
    query_embedding=embedding_service.embed_query("What are the benefits of Startup India initiative?")
    #print(f"query embeddings: {query_embedding}")
    end_time = time()
    print(f"Generated query embedding of dimension: {len(query_embedding)}")
    print(f"Query embedding generation took {end_time - start_time:.2f} seconds.")
    
