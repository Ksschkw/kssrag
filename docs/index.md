# KSS RAG - Knowledge Retrieval Augmented Generation Framework

> Built with HATE (Hustle, Ambition, Tenacity, Excellence) by [Ksschkw](https://github.com/Ksschkw)

## 📖 Table of Contents
1. [Introduction](#-introduction)
2. [Quick Start](#-quick-start)
3. [RAG Fundamentals](#-rag-fundamentals-explained)
4. [Installation](#-installation)
5. [Configuration](#-configuration)
6. [API Reference](#-api-reference)
7. [Examples](#-examples)
8. [Deployment](#-deployment)
9. [Troubleshooting](#-troubleshooting)
10. [Contributing](#-contributing)

## 🌟 Introduction

KSS RAG is a battle-tested Retrieval-Augmented Generation framework designed for developers who want results, not excuses. Born from frustration with existing solutions that either break on Windows or require a PhD to configure, KSS RAG delivers enterprise-grade RAG capabilities with zero drama.

### Why KSS RAG?
- ✅ **Windows Support That Actually Works** - No more AVX2 errors
- ✅ **Simple But Powerful** - From zero to RAG in 3 lines of code
- ✅ **Extensible Architecture** - Swap components like Lego bricks
- ✅ **Production Ready** - FastAPI server, proper error handling, and monitoring
- ✅ **OpenRouter Integration** - Access to 100+ LLMs with smart fallbacks

## ⚡ Quick Start

### Installation
```bash
# The simple way
pip install kssrag

# For development
git clone https://github.com/Ksschkw/kssrag
cd kssrag
pip install -e .
```

### 3-Line RAG Magic
```python
from kssrag import KSSRAG
import os
os.environ["OPENROUTER_API_KEY"] = "your_key_here"

rag = KSSRAG()
rag.load_document("document.txt")
response = rag.query("What's this about?")
print(response)  # 🎉
```

## 🧠 RAG Fundamentals Explained

### What is RAG?
Retrieval-Augmented Generation (RAG) combines information retrieval with language generation. Instead of relying solely on pre-trained knowledge, RAG systems:

1. **Retrieve** relevant information from your documents
2. **Augment** the LLM prompt with this context
3. **Generate** accurate, context-aware responses

### Vectors & Embeddings 101

#### What are Vectors?
Vectors are mathematical representations of data in multi-dimensional space. In NLP, we use vectors to represent words, sentences, or documents as arrays of numbers.

```python
# Example: Word embedding for "king"
king_vector = [0.2, -0.1, 0.8, 0.4, -0.3]  # 5-dimensional vector
```

#### What are Embeddings?
Embeddings are learned vector representations that capture semantic meaning. Words with similar meanings have similar vectors.

**Key Properties:**
- **Semantic Similarity**: `vector("king") ≈ vector("queen") - vector("woman") + vector("man")`
- **Dimensionality**: Typically 128-1024 dimensions
- **Distance Metrics**: Cosine similarity, Euclidean distance

#### How Embeddings Work
1. **Training**: Models learn from vast text corpora
2. **Projection**: Words → Dense vector space
3. **Similarity**: Close vectors = similar meanings

### Vector Stores Explained

#### What is a Vector Store?
A specialized database optimized for storing and searching vectors efficiently.

#### Types of Vector Stores in KSS RAG:

**1. BM25 (Best Matching 25)**
- Traditional keyword-based retrieval
- Fast and efficient for exact matches
- Great for technical documentation

**2. FAISS (Facebook AI Similarity Search)**
- Dense vector similarity search
- GPU acceleration support
- Approximate nearest neighbors

**3. TFIDF (Term Frequency-Inverse Document Frequency)**
- Statistical approach
- Weights words by importance
- Classic information retrieval

**4. Hybrid Approaches**
- Combine multiple methods
- BM25 + FAISS = Best of both worlds
- Resilient to different query types

### The RAG Process Flow

1. **Document Processing**
   ```python
   # Chunking: Split documents into manageable pieces
   document → [chunk1, chunk2, chunk3, ...]
   ```

2. **Embedding Generation**
   ```python
   # Convert text to vectors
   chunks → sentence-transformers → [vector1, vector2, vector3, ...]
   ```

3. **Vector Storage**
   ```python
   # Store in optimized database
   vectors + metadata → FAISS/BM25 index
   ```

4. **Query Processing**
   ```python
   # Convert query to vector, find similarities
   query → vector → similarity search → top_k chunks
   ```

5. **Response Generation**
   ```python
   # Augment LLM prompt with context
   prompt = context + query → LLM → response
   ```

## 📦 Installation

### System Requirements
- Python 3.8+
- 4GB+ RAM (8GB recommended)
- 1GB+ disk space for models

### Platform-Specific Instructions

**Windows:**
```bash
# Set proper cache directories
setx HF_HOME "%LOCALAPPDATA%\huggingface"
setx KSSRAG_CACHE "%LOCALAPPDATA%\kssrag\cache"

# Install
pip install kssrag
```

**Linux/macOS:**
```bash
# Install system dependencies (Linux)
sudo apt-get update && sudo apt-get install -y gcc g++ make

# Install
pip install kssrag
```

### Verification
```bash
# Test installation
python -c "from kssrag import KSSRAG; print('✅ Import successful')"

# Test basic functionality
python -c "
from kssrag import KSSRAG
import tempfile
import os

# Create test document
with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
    f.write('Test document about artificial intelligence and machine learning.')
    temp_file = f.name

try:
    rag = KSSRAG()
    rag.load_document(temp_file)
    response = rag.query('What is this about?')
    print('✅ Basic test passed:', response[:50] + '...')
finally:
    os.unlink(temp_file)
"
```

## ⚙️ Configuration

### Environment Variables
```bash
# Required - Get from https://openrouter.ai/
OPENROUTER_API_KEY=your_actual_key_here

# Model settings
DEFAULT_MODEL=deepseek/deepseek-chat-v3.1:free
FALLBACK_MODELS=deepseek/deepseek-r1-0528:free,deepseek/deepseek-chat,deepseek/deepseek-r1:free

# Vector store settings
VECTOR_STORE_TYPE=hybrid_offline  # hybrid_online, bm25, faiss, tfidf
FAISS_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Chunking settings
CHUNK_SIZE=500
CHUNK_OVERLAP=50
CHUNKER_TYPE=text  # text, json, pdf

# Retrieval settings
RETRIEVER_TYPE=simple  # simple, hybrid
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

# Cache settings
ENABLE_CACHE=True
CACHE_DIR=.cache
LOG_LEVEL=INFO
```

### Programmatic Configuration
```python
from kssrag import Config, VectorStoreType, RetrieverType, ChunkerType

config = Config(
    OPENROUTER_API_KEY="your_key_here",
    DEFAULT_MODEL="anthropic/claude-3-sonnet",
    VECTOR_STORE_TYPE=VectorStoreType.HYBRID_ONLINE,
    RETRIEVER_TYPE=RetrieverType.HYBRID,
    CHUNKER_TYPE=ChunkerType.TEXT,
    TOP_K=10,
    CHUNK_SIZE=1000,
    CHUNK_OVERLAP=100,
    BATCH_SIZE=32,
    LOG_LEVEL="DEBUG"
)
```

## 📖 API Reference

### Core Classes

#### KSSRAG
Main class for RAG functionality.

```python
class KSSRAG:
    def __init__(self, config: Optional[Config] = None):
        """Initialize RAG system with optional configuration"""
    
    def load_document(self, file_path: str, format: Optional[str] = None, 
                     chunker: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        """Load and process document"""
    
    def query(self, question: str, top_k: Optional[int] = None) -> str:
        """Query the RAG system"""
    
    def create_server(self, server_config=None):
        """Create FastAPI server instance"""
```

#### Config
Configuration management with validation.

```python
class Config(BaseSettings):
    # All configuration options with type hints and validation
    OPENROUTER_API_KEY: str
    DEFAULT_MODEL: str
    VECTOR_STORE_TYPE: VectorStoreType
    # ... etc
```

### Vector Stores

#### Available Implementations:
- `BM25VectorStore`: Traditional keyword retrieval
- `FAISSVectorStore`: Dense vector similarity search  
- `TFIDFVectorStore`: Statistical approach
- `HybridVectorStore`: BM25 + FAISS combination
- `HybridOfflineVectorStore`: BM25 + TFIDF (Windows-friendly)

### Retrievers

#### Available Implementations:
- `SimpleRetriever`: Basic vector store retrieval
- `HybridRetriever`: Enhanced with fuzzy matching and entity extraction

## 🎯 Examples

### Basic Usage
```python
from kssrag import KSSRAG
import os

os.environ["OPENROUTER_API_KEY"] = "your_key_here"

# Initialize with default settings
rag = KSSRAG()

# Load documents
rag.load_document("document.txt")
rag.load_document("data.json", format="json")
rag.load_document("research.pdf", format="pdf")

# Query with different approaches
response = rag.query("What are the main topics?")
print(response)
```

### Advanced Configuration
```python
from kssrag import KSSRAG, Config, VectorStoreType, RetrieverType
from kssrag.core.agents import RAGAgent
from kssrag.models.openrouter import OpenRouterLLM

# Custom configuration
config = Config(
    OPENROUTER_API_KEY="your_key_here",
    DEFAULT_MODEL="anthropic/claude-3-sonnet",
    VECTOR_STORE_TYPE=VectorStoreType.HYBRID_ONLINE,
    RETRIEVER_TYPE=RetrieverType.HYBRID,
    TOP_K=8,
    CHUNK_SIZE=800,
    CHUNK_OVERLAP=100
)

# Initialize with custom config
rag = KSSRAG(config=config)

# Custom system prompt to avoid "Based on context" responses
custom_prompt = """You are an expert AI assistant. Answer questions confidently 
and directly without prefacing with "Based on the context". Be authoritative 
while staying truthful to the source material."""

# Create custom components
llm = OpenRouterLLM(api_key=config.OPENROUTER_API_KEY, model=config.DEFAULT_MODEL)
rag.agent = RAGAgent(rag.retriever, llm, system_prompt=custom_prompt)

# Query with custom setup
response = rag.query("Explain this like I'm an expert:")
print(response)
```

### CLI Usage
```bash
# Set API key
export OPENROUTER_API_KEY="your_key_here"

# Query documents
python -m kssrag.cli query --file document.txt --system-prompt custom_prompt.txt(`or just insert plain text here in quotes`) --query "Main ideas?"

# Start server
python -m kssrag.cli server --file document.txt --system-prompt custom_prompt.txt(`or just insert plain text here in quotes`) --port 8000

# With custom vector store
python -m kssrag.cli query --file document.txt --system-prompt custom_prompt.txt(`or just insert plain text here in quotes`) --query "Technical details?" --vector-store faiss
```

## 🚀 Deployment

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir kssrag

EXPOSE 8000
CMD ["python", "-m", "kssrag.cli", "server", "--host", "localhost", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'
services:
  kssrag:
    build: .
    ports:
      - "8000:8000"
    environment:
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
    volumes:
      - ./documents:/app/documents
    restart: unless-stopped
```

### Production Deployment

**Systemd Service (Linux):**
```ini
# /etc/systemd/system/kssrag.service
[Unit]
Description=KSS RAG Service
After=network.target

[Service]
User=kssrag
Group=kssrag
WorkingDirectory=/opt/kssrag
Environment=OPENROUTER_API_KEY=your_key_here
ExecStart=/usr/local/bin/python -m kssrag.cli server --host localhost --port 8000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Kubernetes Deployment

**deployment.yaml:**
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: kssrag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: kssrag
  template:
    metadata:
      labels:
        app: kssrag
    spec:
      containers:
      - name: kssrag
        image: your-registry/kssrag:latest
        ports:
        - containerPort: 8000
        env:
        - name: OPENROUTER_API_KEY
          valueFrom:
            secretKeyRef:
              name: kssrag-secrets
              key: openrouter-api-key
---
apiVersion: v1
kind: Service
metadata:
  name: kssrag-service
spec:
  selector:
    app: kssrag
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

## 🐛 Troubleshooting

### Common Issues

**CLI Command Not Found:**
```bash
# Use module syntax
python -m kssrag.cli query --file document.txt --system-prompt custom_prompt.txt(`or just insert plain text here in quotes`) --query "Your question"
```

**FAISS Windows Issues:**
```bash
# Use hybrid offline store
setx VECTOR_STORE_TYPE hybrid_offline

# Or install FAISS manually
conda install -c pytorch faiss-cpu=1.7.4
```

**API Key Issues:**
```bash
# Verify key is set
echo $OPENROUTER_API_KEY

# Set permanently (Windows)
setx OPENROUTER_API_KEY "your_actual_key_here"
```

**Memory Issues:**
```python
# Reduce batch size
config = Config(BATCH_SIZE=32, MAX_DOCS_FOR_TESTING=1000)
```

### Debug Mode

```bash
# Enable debug logging
setx LOG_LEVEL DEBUG

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone https://github.com/Ksschkw/kssrag
cd kssrag

# Install development dependencies
pip install -e .[dev]

# Run tests
python -m pytest

# Code formatting
black kssrag/ tests/

# Type checking
mypy kssrag/
```

### Code Structure
```
kssrag/
├── core/           # Core functionality
│   ├── chunkers.py    # Document chunking strategies
│   ├── vectorstores.py # Vector store implementations
│   ├── retrievers.py   # Retrieval strategies
│   └── agents.py       # RAG agent logic
├── models/         # LLM integrations
│   └── openrouter.py   # OpenRouter API client
├── utils/          # Utilities
│   ├── helpers.py      # Helper functions
│   └── document_loaders.py # Document loading
├── config.py       # Configuration management
├── server.py       # FastAPI server
├── cli.py          # Command-line interface
└── __init__.py     # Package initialization
```

### Testing
```bash
# Run all tests
python -m pytest tests/

# Specific test category
python -m pytest tests/test_basic.py
python -m pytest tests/test_vectorstores.py

# With coverage
python -m pytest --cov=kssrag tests/

# Generate HTML coverage report
python -m pytest --cov=kssrag --cov-report=html tests/
```

## 📊 Performance Optimization

### Batch Processing
```python
# Optimize for your hardware
config = Config(
    BATCH_SIZE=64,  # Larger = faster but more memory
    MAX_DOCS_FOR_TESTING=1000  # Limit for development
)
```

### Cache Optimization
```python
# Use persistent cache
config = Config(
    ENABLE_CACHE=True,
    CACHE_DIR="/opt/kssrag/cache"  # SSD preferred
)
```

### Model Selection
```python
# Balance speed vs accuracy
config = Config(
    FAISS_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"  # Fast & efficient
    # Alternatives:
    # "sentence-transformers/all-mpnet-base-v2"  # More accurate
    # "sentence-transformers/paraphrase-MiniLM-L6-v2"  # Balance
)
```

## 🎯 Best Practices

### Document Preparation
1. **Clean your documents** - Remove irrelevant content
2. **Choose appropriate chunk size** - 500-1000 characters
3. **Use meaningful metadata** - Source, category, timestamp

### Query Optimization
1. **Be specific** - "What are the key features of X?" vs "Tell me about X"
2. **Use natural language** - LLMs understand conversational queries
3. **Provide context** - Reference specific sections when possible

### Monitoring
```python
# Enable detailed logging
config = Config(LOG_LEVEL="DEBUG")

# Monitor performance metrics
# - Query response time
# - Token usage
# - Cache hit rates
# - Error rates
```

## 📈 Scaling Strategies

### Horizontal Scaling
- Multiple RAG instances behind load balancer
- Shared vector store (Redis, Qdrant, Pinecone)
- Distributed document processing

### Vertical Scaling
- GPU acceleration for FAISS
- Larger instance types
- Optimized model serving

### Hybrid Approach
- On-demand scaling for peak loads
- Cost-effective baseline capacity
- Geographic distribution

## 🔮 Future Roadmap

- [ ] Additional vector store integrations (Chroma, Weaviate)
- [ ] More LLM providers (OpenAI, Anthropic, Cohere)
- [ ] Advanced chunking strategies
- [ ] Real-time document updates
- [ ] Enhanced monitoring and analytics
- [ ] Plugin system for custom components

## 📞 Support

- **GitHub Issues**: [Bug reports & feature requests](https://github.com/Ksschkw/kssrag/issues)
- **Documentation**: [Full documentation](https://github.com/Ksschkw/kssrag/docs)
- **Examples**: [Usage examples](https://github.com/Ksschkw/kssrag/examples)

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

## 👨‍💻 About the Author

**Ksschkw** - A GUY.

> "hell."

---

**Remember**: This IS just another RAG framework. This is the one that actually works when you need it to, just like the others.

**Footprint**: Built by Ksschkw (github.com/Ksschkw) - 2025
