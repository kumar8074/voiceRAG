# ===================================================================================
# Project: VoiceRAG
# File: src/config.py
# Description: Centralized configuration management
# Author: LALAN KUMAR
# Created: [02-03-2026]
# Updated: [04-03-2026]
# LAST MODIFIED BY: LALAN KUMAR  [https://github.com/kumar8074]
# Version: 1.0.0
# ===================================================================================

import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from langchain_openai import ChatOpenAI
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
load_dotenv()

# Set Sarvam API Key
os.environ["OPENAI_API_KEY"] = os.getenv("SARVAM_API_KEY")
os.environ["HF_TOKEN"]= os.getenv("HF_TOKEN")

class ChunkerConfig(BaseSettings):
    chunk_size: int = Field(default=800, description="The size of each text chunk.")
    chunk_overlap: int = Field(default=100, description="The number of overlapping characters between chunks.")
    
    model_config = SettingsConfigDict(
        env_prefix="CHUNKER_",
        case_sensitive=False,
        extra="ignore"
    )

class LLMConfig(BaseSettings):
    model: str = Field(default="sarvam-m", description="The name of the LLM model to use.")
    streaming: bool = Field(default=True, description="stream responses.")
    openai_api_base: str = Field(default="https://api.sarvam.ai/v1", description="Base URL for the Open API.")
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        case_sensitive=False,
        extra="ignore"
    )
    
class EmbeddingConfig(BaseSettings):
    model: str = Field(default="intfloat/multilingual-e5-large", description="The name of the embedding model to use.")
    provider: str = Field(default="hf-inference", description="The provider for the embedding model (e.g., 'hf-inference', 'openai').")
    embedding_dim: int = Field(default=1024, description="The dimensionality of the embedding vectors.")
    batch_size: int = Field(default=32, description="The batch size for embedding generation.")
    
    model_config = SettingsConfigDict(
        env_prefix="EMBEDDING_",
        case_sensitive=False,
        extra="ignore"
    )
       
class QdrantConfig(BaseSettings):
    """Qdrant configuration"""
    host: str = Field(default="localhost")
    port: int = Field(default=6333)
    collection_name: str = Field(default="rag-collection")
    grpc_port: int | None = Field(default=None)
    prefer_grpc: bool = Field(default=False)
    timeout: int = Field(default=10)
    
    model_config=SettingsConfigDict(
        env_prefix="QDRANT_",
        case_sensitive=False,
        extra="ignore"
    )
  
# Initialize configurations    
chunker_config = ChunkerConfig()
llm_config = LLMConfig()
embedding_config = EmbeddingConfig()
qdrant_config = QdrantConfig()


# Initialize models
LLM_MODEL = ChatOpenAI(
    model=llm_config.model,
    streaming=llm_config.streaming,
    openai_api_base=llm_config.openai_api_base,
)


EMBEDDING_MODEL = InferenceClient(
    model=embedding_config.model,
    provider=embedding_config.provider
)

# Export for backward compatibility
CHUNKER_CHUNK_SIZE = chunker_config.chunk_size
CHUNKER_CHUNK_OVERLAP = chunker_config.chunk_overlap
EMBEDDING_DIMENSION = embedding_config.embedding_dim
EMBEDDING_BATCH_SIZE = embedding_config.batch_size
QDRANT_HOST=qdrant_config.host
QDRANT_PORT=qdrant_config.port
QDRANT_COLLECTION_NAME=qdrant_config.collection_name
QDRANT_GRPC_PORT=qdrant_config.grpc_port
QDRANT_PREFER_GRPC=qdrant_config.prefer_grpc
QDRANT_TIMEOUT=qdrant_config.timeout