# Configuration Guide

## Overview

KSS RAG provides extensive configuration options to tailor the framework for your specific use cases. This guide covers all configuration methods, options, and best practices.

## Configuration Methods

### Environment Variables

Create a `.env` file in your project root:

```bash
# Required - OpenRouter API Configuration
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Model Configuration
DEFAULT_MODEL=anthropic/claude-3-sonnet
FALLBACK_MODELS=deepseek/deepseek-chat-v3.1:free,google/gemini-pro-1.5,meta-llama/llama-3-70b-instruct

# Document Processing
CHUNK_SIZE=800
CHUNK_OVERLAP=100
CHUNKER_TYPE=text

# Vector Store Configuration
VECTOR_STORE_TYPE=hybrid_online
FAISS_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Retrieval Configuration
RETRIEVER_TYPE=hybrid
TOP_K=8
FUZZY_MATCH_THRESHOLD=85

# Performance Configuration
BATCH_SIZE=64
MAX_DOCS_FOR_TESTING=
ENABLE_CACHE=true
CACHE_DIR=.rag_cache

# Server Configuration
SERVER_HOST=localhost
SERVER_PORT=8000
CORS_ORIGINS=*
CORS_ALLOW_CREDENTIALS=true
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=Content-Type,Authorization

# OCR Configuration
OCR_DEFAULT_MODE=typed
ENABLE_STREAMING=true

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

### API Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `OPENROUTER_API_KEY` | string | Required | Your OpenRouter API key |
| `DEFAULT_MODEL` | string | `anthropic/claude-3-sonnet` | Primary LLM model |
| `FALLBACK_MODELS` | List[string] | Multiple fallbacks | Models to try if primary fails |

**Example Models:**
```python
# Premium models (higher cost, better quality)
DEFAULT_MODEL="anthropic/claude-3-sonnet"
DEFAULT_MODEL="openai/gpt-4"

# Balanced models
DEFAULT_MODEL="google/gemini-pro-1.5"
DEFAULT_MODEL="meta-llama/llama-3-70b-instruct"

# Cost-effective models
DEFAULT_MODEL="deepseek/deepseek-chat-v3.1:free"
DEFAULT_MODEL="microsoft/wizardlm-2-8x22b"
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
| `ENABLE_CACHE` | bool | true | Enable vector store caching |
| `CACHE_DIR` | string | `.cache` | Cache directory path |
| `MAX_DOCS_FOR_TESTING` | int | None | Limit documents for testing |

### Server Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `SERVER_HOST` | string | `localhost` | Server bind address |
| `SERVER_PORT` | int | 8000 | Server port |
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
