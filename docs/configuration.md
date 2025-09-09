
# Configuration Options

KSS RAG offers extensive configuration options through environment variables or programmatic configuration.

## Configuration Methods

### Environment Variables

Create a `.env` file in your project directory:

```bash
# OpenRouter API Key (Required)
OPENROUTER_API_KEY=your_openrouter_api_key_here

# Model settings
DEFAULT_MODEL=deepseek/deepseek-chat-v3.1:free
FALLBACK_MODELS=deepseek/deepseek-r1-0528:free,deepseek/deepseek-chat,deepseek/deepseek-r1:free

# Chunking settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHUNKER_TYPE=text

# Vector store settings
VECTOR_STORE_TYPE=hybrid_online
FAISS_MODEL_NAME=all-MiniLM-L6-v2

# Retrieval settings
RETRIEVER_TYPE=simple
TOP_K=5
FUZZY_MATCH_THRESHOLD=80

# Performance settings
BATCH_SIZE=64
MAX_DOCS_FOR_TESTING=

# Server settings
SERVER_HOST=localhost
SERVER_PORT=8000
CORS_ORIGINS=*
CORS_ALLOW_CREDENTIALS=True
CORS_ALLOW_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_ALLOW_HEADERS=Content-Type,Authorization

# Advanced settings
ENABLE_CACHE=True
CACHE_DIR=.cache
LOG_LEVEL=INFO

# Custom components (for advanced users)
# CUSTOM_CHUNKER=my_module.MyCustomChunker
# CUSTOM_VECTOR_STORE=my_module.MyCustomVectorStore
# CUSTOM_RETRIEVER=my_module.MyCustomRetriever
# CUSTOM_LLM=my_module.MyCustomLLM
```

### Programmatic Configuration

```python
from kssrag import Config, VectorStoreType, RetrieverType

config = Config(
    OPENROUTER_API_KEY="your_api_key",
    DEFAULT_MODEL="anthropic/claude-3-sonnet",
    VECTOR_STORE_TYPE=VectorStoreType.HYBRID_ONLINE,
    RETRIEVER_TYPE=RetrieverType.HYBRID,
    TOP_K=10,
    CHUNK_SIZE=1000,
    CHUNK_OVERLAP=100,
    CORS_ORIGINS=["https://example.com", "https://api.example.com"]
)
```

## Configuration Options Reference

### OpenRouter Settings

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `DEFAULT_MODEL`: Default model to use for LLM responses
- `FALLBACK_MODELS`: List of fallback models to try if the default model fails

### Chunking Settings

- `CHUNK_SIZE`: Size of text chunks in characters (default: 500)
- `CHUNK_OVERLAP`: Overlap between chunks in characters (default: 50)
- `CHUNKER_TYPE`: Type of chunker to use (text, json, pdf)

### Vector Store Settings

- `VECTOR_STORE_TYPE`: Type of vector store to use
  - `bm25`: BM25 keyword-based retrieval
  - `faiss`: FAISS semantic similarity search
  - `tfidf`: TFIDF vector space model
  - `hybrid_online`: FAISS + BM25 combination
  - `hybrid_offline`: BM25 + TFIDF combination
- `FAISS_MODEL_NAME`: SentenceTransformer model name for FAISS embeddings

### Retrieval Settings

- `RETRIEVER_TYPE`: Type of retriever to use
  - `simple`: Simple retriever using only vector store
  - `hybrid`: Hybrid retriever with fuzzy matching
- `TOP_K`: Number of results to retrieve (default: 5)
- `FUZZY_MATCH_THRESHOLD`: Threshold for fuzzy matching (0-100, default: 80)

### Performance Settings

- `BATCH_SIZE`: Batch size for processing documents (default: 64)
- `MAX_DOCS_FOR_TESTING`: Limit documents for testing (None for all)

### Server Settings

- `SERVER_HOST`: Host to run the server on (default: localhost)
- `SERVER_PORT`: Port to run the server on (default: 8000)
- `CORS_ORIGINS`: List of CORS origins (default: ["*"])
- `CORS_ALLOW_CREDENTIALS`: Whether to allow CORS credentials (default: True)
- `CORS_ALLOW_METHODS`: List of allowed CORS methods
- `CORS_ALLOW_HEADERS`: List of allowed CORS headers

### Advanced Settings

- `ENABLE_CACHE`: Whether to enable caching for vector stores (default: True)
- `CACHE_DIR`: Directory to store cache files (default: .cache)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

### Custom Components

- `CUSTOM_CHUNKER`: Import path to a custom chunker class
- `CUSTOM_VECTOR_STORE`: Import path to a custom vector store class
- `CUSTOM_RETRIEVER`: Import path to a custom retriever class
- `CUSTOM_LLM`: Import path to a custom LLM class

## Configuration Precedence

Configuration values are loaded in the following order of precedence:

1. Programmatic configuration (highest priority)
2. Environment variables
3. Default values (lowest priority)

## Best Practices

1. **Use environment variables for sensitive data** like API keys
2. **Adjust chunk size based on your content**: Smaller chunks for precise retrieval, larger chunks for context preservation
3. **Choose the right vector store** for your use case:
   - BM25: Best for keyword-based retrieval
   - FAISS: Best for semantic similarity
   - Hybrid: Best for balanced performance
4. **Monitor performance** and adjust batch sizes as needed
5. **Use caching** for production deployments to improve performance

## Advanced Configuration

### Custom Components

You can extend KSS RAG with custom components:

```python
# my_custom_chunker.py
from kssrag.core.chunkers import BaseChunker

class MyCustomChunker(BaseChunker):
    def chunk(self, content, metadata=None):
        # Custom chunking logic
        return chunks

# my_app.py
from kssrag import KSSRAG, Config

config = Config(
    CUSTOM_CHUNKER="my_custom_chunker.MyCustomChunker"
)

rag = KSSRAG(config=config)
```

### Dynamic Configuration

You can dynamically update configuration at runtime:

```python
from kssrag import KSSRAG, Config

# Initial configuration
config = Config(TOP_K=5)
rag = KSSRAG(config=config)

# Update configuration
rag.config.TOP_K = 10
```

This documentation covers all configuration options available in KSS RAG. For more specific use cases, check the examples directory or the API reference.
```