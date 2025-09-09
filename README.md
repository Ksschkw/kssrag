# 🚀 KSS RAG - Knowledge Retrieval Augmented Generation Framework

> Built by [Ksschkw](https://github.com/Ksschkw)
\
> View at: [PyPi](https://pypi.org/project/kssrag/0.1.0/)

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.1.0-lightgrey)
![Windows Support](https://img.shields.io/badge/Windows-Friendly-success)
![Docker Ready](https://img.shields.io/badge/Docker-Ready-blue)

**The RAG framework that actually works on your machine - no excuses, no compromises.** 😎

## ✨ Why KSS RAG?

I built this because I was tired of:
- Using thesame codebase over and over 🤬 (I was not really tired, i was just bored)
- Dependencies that require PhD-level installation skills 🎓(This is just buzz)
- Documentation that assumes you already know everything 🤦 (Read a flipping book g)
- APIs that make simple things complicated  - I MIGHT HATE GROQ (No shade)
- I built this for my personal use but you can `pip install kssrag` if you like

## 🚀 Quick Start

### Installation

```bash
# pip installation
pip install kssrag

# or from source
git clone https://github.com/Ksschkw/kssrag
cd kssrag
pip install -e .
```

### Basic Usage

```python
from kssrag import KSSRAG
import os

os.environ["OPENROUTER_API_KEY"] = "your_key_here_do_not_share_this_you_little_dev"

rag = KSSRAG()
rag.load_document("document.txt")
response = rag.query("What's this about?")
print(response)
```

### CLI Usage

```bash
# Set your API key
export OPENROUTER_API_KEY="your_key_here"

# Query documents
python -m kssrag.cli query --file document.txt --query "Main ideas?"

# Start API server
python -m kssrag.cli server --file document.txt --port 8000
```

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
# Create environment file
echo "OPENROUTER_API_KEY=your_key_here" > .env

# Start services
docker-compose up -d

# View logs
docker-compose logs -f
```

### Manual Docker Build

```bash
# Build image
docker build -t kssrag .

# Run container
docker run -p 8000:8000 \
  -e OPENROUTER_API_KEY="your_key_here" \
  -v $(pwd)/documents:/app/documents \
  kssrag
```

## ⚙️ Configuration Mastery

### Environment Variables

```bash
# Required
OPENROUTER_API_KEY=your_openrouter_key

# Model settings
DEFAULT_MODEL=anthropic/claude-3-sonnet # This is a premium model incase you get an error, cheap ass
FALLBACK_MODELS=deepseek/deepseek-chat-v3.1:free,google/gemini-pro-1.5

# Vector stores
VECTOR_STORE_TYPE=hybrid_online #Uses FAISS+BM25 # hybrid_offline also available, uses tfidf + bm25

# Chunking
CHUNK_SIZE=800
CHUNK_OVERLAP=100

# Retrieval
TOP_K=8
FUZZY_MATCH_THRESHOLD=85
```

### Advanced Programmatic Configuration

```python
from kssrag import Config, VectorStoreType, RetrieverType

config = Config(
    OPENROUTER_API_KEY="your_key",
    DEFAULT_MODEL="anthropic/claude-3-sonnet",
    VECTOR_STORE_TYPE=VectorStoreType.HYBRID_ONLINE,
    RETRIEVER_TYPE=RetrieverType.HYBRID,
    TOP_K=10,
    CHUNK_SIZE=1000,
    CHUNK_OVERLAP=150
)
```

## 🎯 Advanced Features

### Custom System Prompts

```python
from kssrag.core.agents import RAGAgent
from kssrag.models.openrouter import OpenRouterLLM

custom_prompt = """You are an expert AI assistant. Answer questions confidently 
and directly without prefacing with "Based on the context". Be authoritative 
while staying truthful to the source material."""

llm = OpenRouterLLM(api_key="your_key", model="anthropic/claude-3-sonnet")
agent = RAGAgent(retriever=rag.retriever, llm=llm, system_prompt=custom_prompt)
```

### Multiple Document Types

```python
# Text files
rag.load_document("notes.txt")

# PDF documents  
rag.load_document("research.pdf", format="pdf")

# JSON data
rag.load_document("data.json", format="json")

# With custom metadata
rag.load_document("file.txt", metadata={"source": "internal", "category": "technical"})
```

## 📊 Performance Optimization

### Batch Processing

```python
config = Config(
    BATCH_SIZE=64,  # Larger batches for better performance
    MAX_DOCS_FOR_TESTING=1000  # Limit for testing
)
```

### Cache Management

```python
config = Config(
    ENABLE_CACHE=True,
    CACHE_DIR="./.rag_cache",  # Custom cache location ---DO NOT DO THIS IF YOU ARE USIINF FAISS ON WINDOWS, f**k around and find out
    LOG_LEVEL="DEBUG"  # Detailed logging
)
```

## 🧪 Testing & Validation

```bash
# Run all tests
python -m pytest tests/

# Specific test file
python -m pytest tests/test_basic.py

# With coverage report
python -m pytest --cov=kssrag tests/
```

## 🚨 Troubleshooting

### Common Issues

**CLI Command Not Found**
```bash
# Use module syntax on Windows
python -m kssrag.cli query --file document.txt --query "Your question"
```

**FAISS Windows Issues**
```bash
# Use hybrid offline vector store
setx VECTOR_STORE_TYPE hybrid_offline
```

**API Key Issues**
```bash
# Verify your OpenRouter key
echo $OPENROUTER_API_KEY

# Or set it permanently
setx OPENROUTER_API_KEY "your_actual_key_here"
```

### Debug Mode

```bash
# Enable debug logging
setx LOG_LEVEL DEBUG

# Or in code
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Production Deployment

### Environment Setup

```bash
# Create production environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# Install with production dependencies
pip install kssrag
```

### Systemd Service (Linux)

```ini
#I do not know this, this is AI slop i do not know if it works, tho it might
# /etc/systemd/system/kssrag.service
[Unit]
Description=KSS RAG Service
After=network.target

[Service]
User=appuser
Group=appuser
WorkingDirectory=/opt/kssrag
Environment=OPENROUTER_API_KEY=your_key_here
ExecStart=/opt/kssrag/venv/bin/python -m kssrag.cli server --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
```

## 🤝 Contributing

### Development Setup

```bash
# Clone and setup
git clone https://github.com/Ksschkw/kssrag
cd kssrag

# Install development dependencies
pip install -e .[dev]

# Run tests
python -m pytest

# Code formatting
black kssrag/ tests/
```

### Code Structure

```
kssrag/
├── core/           # Core functionality
├── models/         # LLM integrations
├── utils/          # Utilities & helpers
├── config.py       # Configuration management
├── server.py       # FastAPI server
└── cli.py          # Command-line interface
```

## 📚 Learning Resources

### RAG Fundamentals

1. **Vectors**: Numerical representations of text
2. **Embeddings**: Dense vector representations capturing semantic meaning  
3. **Vector Stores**: Databases optimized for vector similarity search
4. **Retrieval**: Finding relevant context for questions
5. **Generation**: Creating responses using retrieved context

### Next Steps

1. Experiment with different vector store types
2. Try various chunking strategies
3. Customize the system prompt for your use case
4. Explore different LLM models on OpenRouter
5. Monitor and optimize performance

## 🏆 Success Stories

> "NEIN 

## 📞 Support

- **GitHub Issues**: [Bug reports & feature requests](https://github.com/Ksschkw/kssrag/issues)
- **Documentation**: [Full documentation](docs/)
- **Examples**: [Usage examples](examples/)

## 📜 License

MIT License - do whatever you want, just don't be a vigil (I hate vigils).

## 👨‍💻 About the Author

**Ksschkw** - Just check my github page g.

> "Built with HATE"

---

**Remember**: This IS just another RAG framework. This is the one that actually works when you need it to. 🚀 \
(Yes that was NOT a typo)

**Footprint**: Built with HATE by Ksschkw (github.com/Ksschkw) - 2025
```