# Configuration Guide

## Overview

KSS RAG provides extensive configuration options to tailor the framework for your specific use cases. This guide covers all configuration methods, options, and best practices.

## Configuration Methods

### Environment Variables

Create a `.env` file in your project root:

```bash
# LLM Provider (see the LLM Providers table below)
PROVIDER=openrouter
# API key for the selected provider. If unset, the provider's own env var is used
# (GROQ_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, ...), falling back to OPENROUTER_API_KEY.
LLM_API_KEY=
# Only needed for PROVIDER=custom, or to override a preset's endpoint:
LLM_BASE_URL=https://openrouter.ai/api/v1/chat/completions

# OpenRouter key (used when PROVIDER=openrouter)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Model Configuration
DEFAULT_MODEL=deepseek/deepseek-chat-v3.1:free
FALLBACK_MODELS=deepseek/deepseek-r1:free,deepseek/deepseek-chat

# LLM generation parameters
LLM_TEMPERATURE=0.7
LLM_MAX_TOKENS=1024
LLM_TIMEOUT=30
LLM_STREAM_TIMEOUT=60

# Document Processing
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHUNKER_TYPE=text

# Vector Store Configuration
VECTOR_STORE_TYPE=hybrid_offline
FAISS_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Retrieval Configuration
RETRIEVER_TYPE=simple
TOP_K=5
FUZZY_MATCH_THRESHOLD=80

# Performance Configuration
BATCH_SIZE=64
MAX_DOCS_FOR_TESTING=
ENABLE_CACHE=true
CACHE_DIR=  # defaults to a per-user cache dir (~/.cache/kssrag)

# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=8000
MAX_SESSIONS=1000
CORS_ORIGINS=*
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=Content-Type,Authorization

# OCR Configuration
OCR_DEFAULT_MODE=typed
ENABLE_STREAMING=false

# Logging Configuration
LOG_LEVEL=INFO
```

### Programmatic Configuration

```python
from kssrag import Config, VectorStoreType, RetrieverType, ChunkerType

config = Config(
    # API Configuration
    OPENROUTER_API_KEY="your-api-key",
    DEFAULT_MODEL="anthropic/claude-3-sonnet",
    FALLBACK_MODELS=[
        "deepseek/deepseek-chat-v3.1:free",
        "google/gemini-pro-1.5"
    ],
    
    # Document Processing
    CHUNK_SIZE=1000,
    CHUNK_OVERLAP=150,
    CHUNKER_TYPE=ChunkerType.TEXT,
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE=VectorStoreType.HYBRID_ONLINE,
    FAISS_MODEL_NAME="sentence-transformers/all-mpnet-base-v2",
    
    # Retrieval Configuration
    RETRIEVER_TYPE=RetrieverType.HYBRID,
    TOP_K=10,
    FUZZY_MATCH_THRESHOLD=80,
    
    # Performance Configuration
    BATCH_SIZE=32,
    ENABLE_CACHE=True,
    CACHE_DIR="/opt/kssrag/cache",
    
    # Server Configuration
    SERVER_HOST="0.0.0.0",
    SERVER_PORT=8080,
    CORS_ORIGINS=["https://app.example.com"],
    
    # Advanced Features
    OCR_DEFAULT_MODE="typed",
    ENABLE_STREAMING=True,
    LOG_LEVEL="DEBUG"
)
```

## Configuration Reference

### LLM Provider Configuration

KSS RAG talks to any LLM provider through a single `PROVIDER` setting (or the
`--provider` CLI flag). OpenRouter is the default.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `PROVIDER` | string | `openrouter` | Provider preset name, or `custom` |
| `LLM_API_KEY` | string | None | Key for the selected provider (see resolution below) |
| `LLM_BASE_URL` | string | OpenRouter URL | Endpoint override (required for `custom`) |
| `OPENROUTER_API_KEY` | string | `""` | OpenRouter key / legacy fallback key |
| `DEFAULT_MODEL` | string | `deepseek/deepseek-chat` | Primary model id |
| `FALLBACK_MODELS` | List[string] | DeepSeek models | Models to try if the primary fails |
| `LLM_TEMPERATURE` | float | 0.7 | Sampling temperature (0.0–2.0) |
| `LLM_MAX_TOKENS` | int | 1024 | Max tokens per response |
| `LLM_TIMEOUT` | int | 30 | Non-streaming request timeout (s) |
| `LLM_STREAM_TIMEOUT` | int | 60 | Streaming request timeout (s) |

**Supported providers:**

| Kind | Names |
|------|-------|
| Hosted (OpenAI-compatible) | `openrouter`, `openai`, `groq`, `together`, `deepseek`, `fireworks`, `mistral`, `perplexity`, `xai`, `deepinfra`, `anyscale` |
| Native protocol | `anthropic`, `ollama` |
| Local (no key) | `ollama-openai`, `lmstudio`, `vllm`, `llamacpp` |
| Custom endpoint | `custom` (set `LLM_BASE_URL`) |

**API key resolution order:** explicit `api_key` / `--api-key` → `LLM_API_KEY` →
the provider's own env var (e.g. `GROQ_API_KEY`) → `OPENROUTER_API_KEY`. Local
providers need no key.

**From code:**

```python
from kssrag import create_llm

llm = create_llm(provider="groq", model="llama-3.3-70b-versatile")
llm = create_llm(provider="ollama", model="llama3")            # local, no key
llm = create_llm(provider="anthropic", model="claude-sonnet-4-6")
llm = create_llm(provider="custom",
                 base_url="http://localhost:8000/v1/chat/completions", model="m")
```

### Document Processing Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `CHUNK_SIZE` | int | 800 | 100-2000 | Characters per chunk |
| `CHUNK_OVERLAP` | int | 100 | 0-500 | Character overlap between chunks |
| `CHUNKER_TYPE` | enum | `text` | text,json,pdf,image | Document chunking strategy |

**Chunking Best Practices:**

```python
# Technical documentation
Config(CHUNK_SIZE=600, CHUNK_OVERLAP=75)

# Academic papers
Config(CHUNK_SIZE=1200, CHUNK_OVERLAP=150)

# Conversational content
Config(CHUNK_SIZE=500, CHUNK_OVERLAP=50)

# Code and structured data
Config(CHUNK_SIZE=400, CHUNK_OVERLAP=25)
```

### Vector Store Configuration

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `VECTOR_STORE_TYPE` | enum | `hybrid_offline` | bm25, bm25s, faiss, tfidf, hybrid_online, hybrid_offline | Vector store implementation |
| `FAISS_MODEL_NAME` | string | `all-MiniLM-L6-v2` | Various | Sentence transformer model |

**Vector Store Selection Guide:**

```python
# Keyword-heavy content (documentation, code)
VectorStoreType.BM25
VectorStoreType.BM25S

# Semantic search (research papers, creative content)
VectorStoreType.FAISS

# Balanced approach (enterprise applications)
VectorStoreType.HYBRID_ONLINE

# Resource-constrained environments
VectorStoreType.HYBRID_OFFLINE
```

**FAISS Model Options:**
```python
# Fast and efficient
FAISS_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"

# Higher accuracy
FAISS_MODEL_NAME="sentence-transformers/all-mpnet-base-v2"

# Multilingual support
FAISS_MODEL_NAME="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### Retrieval Configuration

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `TOP_K` | int | 5 | 1-20 | Number of results to retrieve |
| `FUZZY_MATCH_THRESHOLD` | int | 80 | 0-100 | Fuzzy matching sensitivity |
| `RETRIEVER_TYPE` | enum | `simple` | simple, hybrid | Retrieval strategy |

### Performance Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `BATCH_SIZE` | int | 64 | Documents per processing batch |
| `ENABLE_CACHE` | bool | true | Enable index caching (reuse built indexes) |
| `CACHE_DIR` | string | `~/.cache/kssrag` | Cache directory (platform-appropriate default) |
| `MAX_DOCS_FOR_TESTING` | int | None | Limit documents for testing |

### Server Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SERVER_HOST` | string | `localhost` | Server bind address |
| `SERVER_PORT` | int | 8000 | Server port |
| `MAX_SESSIONS` | int | 1000 | Max in-memory conversation sessions (LRU eviction) |
| `CORS_ORIGINS` | List[string] | `["*"]` | Allowed CORS origins |
| `CORS_ALLOW_CREDENTIALS` | bool | true | Allow CORS credentials |
| `CORS_ALLOW_METHODS` | List[string] | Multiple | Allowed HTTP methods |
| `CORS_ALLOW_HEADERS` | List[string] | Multiple | Allowed HTTP headers |

### OCR Configuration

| Parameter | Type | Default | Options | Description |
|-----------|------|---------|---------|-------------|
| `OCR_DEFAULT_MODE` | string | `typed` | typed, handwritten | Default OCR engine |
| `ENABLE_STREAMING` | bool | false | Enable response streaming |

## Advanced Configuration

### Custom Components

```python
# Custom chunker implementation
CUSTOM_CHUNKER="my_package.chunkers.SemanticChunker"

# Custom vector store
CUSTOM_VECTOR_STORE="my_package.vectorstores.ChromaVectorStore"

# Custom retriever
CUSTOM_RETRIEVER="my_package.retrievers.AdvancedRetriever"

# Custom LLM
CUSTOM_LLM="my_package.llms.CustomLLMProvider"
```

### Dynamic Configuration

```python
from kssrag import KSSRAG, Config

# Initial configuration
rag = KSSRAG(config=Config(TOP_K=5))

# Runtime configuration updates
rag.config.TOP_K = 10
rag.config.CHUNK_SIZE = 1000

# Reload with new configuration
rag.vector_store.add_documents(rag.documents)
```

## Configuration Validation

KSS RAG includes comprehensive configuration validation:

```python
try:
    config = Config(
        OPENROUTER_API_KEY="invalid-key",
        CHUNK_SIZE=50  # Below minimum
    )
except ValidationError as e:
    print(f"Configuration error: {e}")

# Valid configuration ranges:
# CHUNK_SIZE: 100-2000
# CHUNK_OVERLAP: 0-500  
# TOP_K: 1-20
# BATCH_SIZE: 1-256
# FUZZY_MATCH_THRESHOLD: 0-100
```

## Best Practices

### Production Configuration

```python
production_config = Config(
    OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
    DEFAULT_MODEL="anthropic/claude-3-sonnet",
    VECTOR_STORE_TYPE=VectorStoreType.HYBRID_ONLINE,
    CHUNK_SIZE=1000,
    TOP_K=8,
    BATCH_SIZE=32,
    ENABLE_CACHE=True,
    CACHE_DIR="/var/lib/kssrag/cache",
    LOG_LEVEL="INFO"
)
```

### Development Configuration

```python
development_config = Config(
    OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
    VECTOR_STORE_TYPE=VectorStoreType.BM25,
    MAX_DOCS_FOR_TESTING=100,
    CHUNK_SIZE=500,
    LOG_LEVEL="DEBUG"
)
```

### Security Considerations

```python
secure_config = Config(
    # Never hardcode API keys
    OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
    
    # Restrict CORS in production
    CORS_ORIGINS=["https://yourdomain.com"],
    CORS_ALLOW_CREDENTIALS=False,
    
    # Secure cache directory
    CACHE_DIR="/secure/path/kssrag/cache"
)
```

## Troubleshooting

### Common Configuration Issues

**API Key Problems:**
```python
# Error: OPENROUTER_API_KEY not set
solution = "Set environment variable or pass in Config"

# Error: Invalid API key
solution = "Verify key validity at https://openrouter.ai/keys"
```

**Vector Store Issues:**
```python
# Error: FAISS not available on Windows
solution = "Use HYBRID_OFFLINE or install faiss-cpu-win"

# Error: Memory issues with large documents
solution = "Reduce BATCH_SIZE and CHUNK_SIZE"
```

**Performance Issues:**
```python
# Slow document processing
solution = "Adjust CHUNK_SIZE and BATCH_SIZE"

# High memory usage  
solution = "Enable caching and reduce MAX_DOCS_FOR_TESTING"
```

This configuration guide provides comprehensive coverage of all KSS RAG configuration options. For specific use cases or advanced scenarios, refer to the API reference or examples directory.

---
