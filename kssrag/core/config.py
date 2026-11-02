"""
Configuration settings and typed feature flag declarations for KSS RAG.
Conforms to house-standard.md §11, §14.
"""
import os
import platform
from enum import Enum
from typing import List, Optional, Dict, Any
from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


def _default_cache_dir() -> str:
    """Platform-appropriate cache directory, overridable via CACHE_DIR env var."""
    env = os.getenv("CACHE_DIR")
    if env:
        return env
    if platform.system() == "Windows":
        base = os.getenv("LOCALAPPDATA", os.path.expanduser("~"))
        return os.path.join(base, ".kssrag", "cache")
    return os.path.join(os.path.expanduser("~"), ".cache", "kssrag")


class VectorStoreType(str, Enum):
    BM25 = "bm25"
    BM25S = "bm25s"
    FAISS = "faiss"
    TFIDF = "tfidf"
    HYBRID_ONLINE = "hybrid_online"
    HYBRID_OFFLINE = "hybrid_offline"
    CUSTOM = "custom"


class ChunkerType(str, Enum):
    TEXT = "text"
    SENTENCE = "sentence"
    JSON = "json"
    PDF = "pdf"
    IMAGE = "image"
    CUSTOM = "custom"


class RetrieverType(str, Enum):
    SIMPLE = "simple"
    HYBRID = "hybrid"
    CUSTOM = "custom"


class Config(BaseSettings):
    """Configuration settings for KSS RAG with typed feature flags and bounds."""

    model_config = SettingsConfigDict(
        case_sensitive=True,
        extra="ignore",
        # Keeps comma-separated lists (FALLBACK_MODELS, CORS_*, FALLBACK_PROVIDERS)
        # working; they are split by the field_validator below.
        enable_decoding=False,
    )

    # -------------------------------------------------------------------------
    # Feature Flags (§14)
    # Temporary runtime toggles. Date added and removal conditions documented.
    # -------------------------------------------------------------------------
    # Date added: 2026-11-02 | Removal condition: Permanent once breaker proven in production
    feature_resilience_circuit_breaker: bool = Field(
        default=os.getenv("FEATURE_RESILIENCE_CIRCUIT_BREAKER", "true").lower() == "true",
        description="Enable circuit breakers on outbound LLM calls."
    )

    # Date added: 2026-11-02 | Removal condition: Next minor release cycle
    feature_search_semantic_ranking: bool = Field(
        default=os.getenv("FEATURE_SEARCH_SEMANTIC_RANKING", "true").lower() == "true",
        description="Enable semantic ranking in hybrid vector stores."
    )

    # Date added: 2026-11-02 | Removal condition: Next minor release cycle
    feature_eval_detailed_metrics: bool = Field(
        default=os.getenv("FEATURE_EVAL_DETAILED_METRICS", "false").lower() == "true",
        description="Emit per-query metric diagnostics during evaluation."
    )

    # -------------------------------------------------------------------------
    # LLM Provider settings
    # -------------------------------------------------------------------------
    OPENROUTER_API_KEY: str = Field(
        default=os.getenv("OPENROUTER_API_KEY", ""),
        description="Your OpenRouter API key for accessing LLMs"
    )

    PROVIDER: str = Field(
        default=os.getenv("PROVIDER", "openrouter"),
        description="LLM provider preset name (openrouter, openai, groq, ollama, anthropic, ...) or 'custom'"
    )

    LLM_BASE_URL: str = Field(
        default=os.getenv("LLM_BASE_URL", "https://openrouter.ai/api/v1/chat/completions"),
        description="Chat completions endpoint URL"
    )

    LLM_API_KEY: Optional[str] = Field(
        default=os.getenv("LLM_API_KEY"),
        description="API key for the selected provider"
    )

    DEFAULT_MODEL: str = Field(
        default=os.getenv("DEFAULT_MODEL", "deepseek/deepseek-chat"),
        description="Default model to use for LLM responses"
    )

    FALLBACK_MODELS: List[str] = Field(
        default=os.getenv("FALLBACK_MODELS", "deepseek/deepseek-r1-0528:free,deepseek/deepseek-chat,deepseek/deepseek-r1:free").split(","),
        description="List of fallback models to try if the default model fails"
    )

    FALLBACK_PROVIDERS: List[str] = Field(
        default=os.getenv("FALLBACK_PROVIDERS", ""),
        description="Comma-separated provider presets tried in order if the primary provider fails"
    )

    LLM_TEMPERATURE: float = Field(
        default=float(os.getenv("LLM_TEMPERATURE", 0.7)),
        ge=0.0,
        le=2.0,
        description="Sampling temperature for LLM responses"
    )

    LLM_MAX_TOKENS: int = Field(
        default=int(os.getenv("LLM_MAX_TOKENS", 1024)),
        ge=1,
        le=32768,
        description="Maximum tokens to generate per LLM response"
    )

    LLM_TIMEOUT: int = Field(
        default=int(os.getenv("LLM_TIMEOUT", 30)),
        ge=1,
        le=600,
        description="Per-request timeout (seconds) for non-streaming LLM calls"
    )

    LLM_STREAM_TIMEOUT: int = Field(
        default=int(os.getenv("LLM_STREAM_TIMEOUT", 60)),
        ge=1,
        le=600,
        description="Per-request timeout (seconds) for streaming LLM calls"
    )

    LLM_MAX_RETRIES: int = Field(
        default=int(os.getenv("LLM_MAX_RETRIES", 3)),
        ge=0,
        le=10,
        description="Times to retry the same model on a 429 rate-limit response"
    )

    LLM_RETRY_BACKOFF: float = Field(
        default=float(os.getenv("LLM_RETRY_BACKOFF", 2.0)),
        ge=0.0,
        le=60.0,
        description="Base backoff (seconds) between retries"
    )

    # -------------------------------------------------------------------------
    # Chunking & Vector Store settings
    # -------------------------------------------------------------------------
    CHUNK_SIZE: int = Field(
        default=int(os.getenv("CHUNK_SIZE", 500)),
        ge=50,
        le=10000,
        description="Size of text chunks in characters"
    )

    CHUNK_OVERLAP: int = Field(
        default=int(os.getenv("CHUNK_OVERLAP", 50)),
        ge=0,
        le=5000,
        description="Overlap between chunks in characters"
    )

    CHUNKER_TYPE: ChunkerType = Field(
        default=os.getenv("CHUNKER_TYPE", ChunkerType.TEXT),
        description="Type of chunker to use"
    )

    VECTOR_STORE_TYPE: VectorStoreType = Field(
        default=os.getenv("VECTOR_STORE_TYPE", VectorStoreType.HYBRID_OFFLINE),
        description="Type of vector store to use"
    )

    FAISS_MODEL_NAME: str = Field(
        default=os.getenv("FAISS_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        description="SentenceTransformer model name for FAISS embeddings"
    )

    RETRIEVER_TYPE: RetrieverType = Field(
        default=os.getenv("RETRIEVER_TYPE", RetrieverType.SIMPLE),
        description="Type of retriever to use"
    )

    TOP_K: int = Field(
        default=int(os.getenv("TOP_K", 5)),
        ge=1,
        le=100,
        description="Number of results to retrieve"
    )

    FUZZY_MATCH_THRESHOLD: int = Field(
        default=int(os.getenv("FUZZY_MATCH_THRESHOLD", 80)),
        ge=0,
        le=100,
        description="Threshold for fuzzy matching (0-100)"
    )

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

    # -------------------------------------------------------------------------
    # Server & Transport settings (§11 Security: secure CORS defaults)
    # -------------------------------------------------------------------------
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

    MAX_SESSIONS: int = Field(
        default=int(os.getenv("MAX_SESSIONS", 1000)),
        ge=1,
        description="Maximum number of concurrent conversation sessions kept in memory (LRU eviction)"
    )

    # House Standard §11: CORS is an explicit allowlist. Never '*' with credentials.
    CORS_ORIGINS: List[str] = Field(
        default=os.getenv("CORS_ORIGINS", "*").split(","),
        description="List of CORS origins"
    )

    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=os.getenv("CORS_ALLOW_CREDENTIALS", "false").lower() == "true",
        description="Whether to allow CORS credentials (default False to prevent security violations with wildcard origins)"
    )

    CORS_ALLOW_METHODS: List[str] = Field(
        default=os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS").split(","),
        description="List of allowed CORS methods"
    )

    CORS_ALLOW_HEADERS: List[str] = Field(
        default=os.getenv("CORS_ALLOW_HEADERS", "Content-Type,Authorization,X-Correlation-ID").split(","),
        description="List of allowed CORS headers"
    )

    RATE_LIMIT_PER_MINUTE: int = Field(
        default=int(os.getenv("RATE_LIMIT_PER_MINUTE", 120)),
        ge=1,
        le=10000,
        description="Rate limit per minute per principal/IP"
    )

    RATE_LIMIT_MAX_TRACKED_CLIENTS: int = Field(
        default=int(os.getenv("RATE_LIMIT_MAX_TRACKED_CLIENTS", 10000)),
        ge=1,
        le=1000000,
        description=(
            "Maximum number of client identities the rate limiter keeps counters for. "
            "Bounds memory on a public endpoint; the least recently seen client is evicted."
        )
    )

    TRUST_FORWARDED_FOR: bool = Field(
        default=os.getenv("TRUST_FORWARDED_FOR", "false").lower() == "true",
        description=(
            "Whether to take the client IP from the first X-Forwarded-For entry. "
            "Leave False unless the service runs behind a trusted proxy that sets the "
            "header: when False the socket peer address is used."
        )
    )

    ENABLE_CACHE: bool = Field(
        default=os.getenv("ENABLE_CACHE", "True").lower() == "true",
        description="Whether to enable caching for vector stores"
    )

    CACHE_DIR: str = Field(
        default_factory=_default_cache_dir,
        description="Directory to store cache files"
    )

    LOG_LEVEL: str = Field(
        default=os.getenv("LOG_LEVEL", "INFO"),
        description="Logging level"
    )

    OCR_DEFAULT_MODE: str = Field(
        default=os.getenv("OCR_DEFAULT_MODE", "typed"),
        description="Default OCR mode (typed or handwritten)"
    )

    # Custom components
    CUSTOM_CHUNKER: Optional[str] = Field(default=os.getenv("CUSTOM_CHUNKER"))
    CUSTOM_VECTOR_STORE: Optional[str] = Field(default=os.getenv("CUSTOM_VECTOR_STORE"))
    CUSTOM_RETRIEVER: Optional[str] = Field(default=os.getenv("CUSTOM_RETRIEVER"))
    CUSTOM_LLM: Optional[str] = Field(default=os.getenv("CUSTOM_LLM"))

    # Streaming settings
    ENABLE_STREAMING: bool = Field(
        default=os.getenv("ENABLE_STREAMING", "False").lower() == "true",
        description="Whether to enable streaming responses"
    )

    @field_validator(
        'FALLBACK_MODELS', 'FALLBACK_PROVIDERS',
        'CORS_ORIGINS', 'CORS_ALLOW_METHODS', 'CORS_ALLOW_HEADERS',
        mode='before',
    )
    @classmethod
    def split_string(cls, v):
        if isinstance(v, str):
            return [item.strip() for item in v.split(',') if item.strip()]
        return v


def log_feature_flags(cfg: Config) -> None:
    """Log state of each feature flag at startup at INFO level (§14 rule 4)."""
    from .logging import logger
    flags = [
        ("FEATURE_RESILIENCE_CIRCUIT_BREAKER", cfg.feature_resilience_circuit_breaker),
        ("FEATURE_SEARCH_SEMANTIC_RANKING", cfg.feature_search_semantic_ranking),
        ("FEATURE_EVAL_DETAILED_METRICS", cfg.feature_eval_detailed_metrics),
    ]
    for name, value in flags:
        status_str = "[OK]" if value else "[SKIP]"
        logger.info(f"{status_str} Feature flag {name}={value}")


config = Config()
log_feature_flags(config)
