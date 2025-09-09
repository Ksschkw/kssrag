import os
from dotenv import load_dotenv
from typing import List, Optional, Dict, Any
from pydantic import Field, validator
from pydantic_settings import BaseSettings
from enum import Enum

load_dotenv()

class VectorStoreType(str, Enum):
    BM25 = "bm25"
    FAISS = "faiss"
    TFIDF = "tfidf"
    HYBRID_ONLINE = "hybrid_online"
    HYBRID_OFFLINE = "hybrid_offline"
    CUSTOM = "custom"

class ChunkerType(str, Enum):
    TEXT = "text"
    JSON = "json"
    PDF = "pdf"
    CUSTOM = "custom"

class RetrieverType(str, Enum):
    SIMPLE = "simple"
    HYBRID = "hybrid"
    CUSTOM = "custom"

class Config(BaseSettings):
    """Configuration settings for KSS RAG with extensive customization options"""
    
    # OpenRouter settings
    OPENROUTER_API_KEY: str = Field(
        default=os.getenv("OPENROUTER_API_KEY", ""),
        description="Your OpenRouter API key for accessing LLMs"
    )
    
    DEFAULT_MODEL: str = Field(
        default=os.getenv("DEFAULT_MODEL", "deepseek/deepseek-chat-v3.1:free"),
        description="Default model to use for LLM responses"
    )
    
    FALLBACK_MODELS: List[str] = Field(
        default=os.getenv("FALLBACK_MODELS", "deepseek/deepseek-r1-0528:free,deepseek/deepseek-chat,deepseek/deepseek-r1:free").split(","),
        description="List of fallback models to try if the default model fails"
    )
    
    # Chunking settings
    CHUNK_SIZE: int = Field(
        default=int(os.getenv("CHUNK_SIZE", 500)),
        ge=100,
        le=2000,
        description="Size of text chunks in characters"
    )
    
    CHUNK_OVERLAP: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", 50)),
        ge=0,
        le=500,
        description="Overlap between chunks in characters"
    )
    
    CHUNKER_TYPE: ChunkerType = Field(
        default=os.getenv("CHUNKER_TYPE", ChunkerType.TEXT),
        description="Type of chunker to use"
    )
    
    # Vector store settings
    VECTOR_STORE_TYPE: VectorStoreType = Field(
        default=os.getenv("VECTOR_STORE_TYPE", VectorStoreType.HYBRID_OFFLINE),
        description="Type of vector store to use"
    )
    
    FAISS_MODEL_NAME: str = Field(
        default=os.getenv("FAISS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        description="SentenceTransformer model name for FAISS embeddings"
    )
    
    # Retrieval settings
    RETRIEVER_TYPE: RetrieverType = Field(
        default=os.getenv("RETRIEVER_TYPE", RetrieverType.SIMPLE),
        description="Type of retriever to use"
    )
    
    TOP_K: int = Field(
        default=int(os.getenv("TOP_K", 5)),
        ge=1,
        le=20,
        description="Number of results to retrieve"
    )
    
    FUZZY_MATCH_THRESHOLD: int = Field(
        default=int(os.getenv("FUZZY_MATCH_THRESHOLD", 80)),
        ge=0,
        le=100,
        description="Threshold for fuzzy matching (0-100)"
    )
    
    # Performance settings
    BATCH_SIZE: int = Field(
        default=int(os.getenv("BATCH_SIZE", 64)),
        ge=1,
        le=256,
        description="Batch size for processing documents"
    )
    
    MAX_DOCS_FOR_TESTING: Optional[int] = Field(
        default=os.getenv("MAX_DOCS_FOR_TESTING"),
        description="Limit documents for testing (None for all)"
    )
    
    # Server settings
    SERVER_HOST: str = Field(
        default=os.getenv("SERVER_HOST", "localhost"),
        description="Host to run the server on"
    )
    
    SERVER_PORT: int = Field(
        default=int(os.getenv("SERVER_PORT", 8000)),
        ge=1024,
        le=65535,
        description="Port to run the server on"
    )
    
    CORS_ORIGINS: List[str] = Field(
        default=os.getenv("CORS_ORIGINS", "*").split(","),
        description="List of CORS origins"
    )
    
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=os.getenv("CORS_ALLOW_CREDENTIALS", "True").lower() == "true",
        description="Whether to allow CORS credentials"
    )
    
    CORS_ALLOW_METHODS: List[str] = Field(
        default=os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(","),
        description="List of allowed CORS methods"
    )
    
    CORS_ALLOW_HEADERS: List[str] = Field(
        default=os.getenv("CORS_ALLOW_HEADERS", "Content-Type,Authorization").split(","),
        description="List of allowed CORS headers"
    )
    
    # Advanced settings
    ENABLE_CACHE: bool = Field(
        default=os.getenv("ENABLE_CACHE", "True").lower() == "true",
        description="Whether to enable caching for vector stores"
    )
    
    CACHE_DIR: str = Field(
        default=os.getenv("CACHE_DIR", ".cache"),
        description="Directory to store cache files"
    )
    
    LOG_LEVEL: str = Field(
        default=os.getenv("LOG_LEVEL", "INFO"),
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    
    # Custom components (for advanced users)
    CUSTOM_CHUNKER: Optional[str] = Field(
        default=os.getenv("CUSTOM_CHUNKER"),
        description="Import path to a custom chunker class"
    )
    
    CUSTOM_VECTOR_STORE: Optional[str] = Field(
        default=os.getenv("CUSTOM_VECTOR_STORE"),
        description="Import path to a custom vector store class"
    )
    
    CUSTOM_RETRIEVER: Optional[str] = Field(
        default=os.getenv("CUSTOM_RETRIEVER"),
        description="Import path to a custom retriever class"
    )
    
    CUSTOM_LLM: Optional[str] = Field(
        default=os.getenv("CUSTOM_LLM"),
        description="Import path to a custom LLM class"
    )
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        use_enum_values = True
        
    @validator('FALLBACK_MODELS', 'CORS_ORIGINS', 'CORS_ALLOW_METHODS', 'CORS_ALLOW_HEADERS', pre=True)
    def split_string(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',')]
        return v

config = Config()