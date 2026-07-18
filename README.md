# KSS RAG

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Version](https://img.shields.io/badge/version-0.2.5-brightgreen)
![Framework](https://img.shields.io/badge/framework-RAG-orange)

**Stand up a streaming RAG API from a single document with one command.**

[Quick Start](#quick-start) • [Why KSS RAG](#why-kss-rag) • [CLI](#cli) • [Python API](#python-api) • [Architecture](#architecture) • [Configuration](#configuration)

</div>

## Overview

KSS RAG is a Retrieval-Augmented Generation framework built around one idea: getting from *"I have a document"* to *"I have a live, streaming Q&A API over it"* should take a single command — no glue code, no notebook, no orchestration boilerplate.

Point the CLI at a source-of-truth document, hand it a system prompt, and it loads, chunks, indexes, and serves a FastAPI endpoint with Server-Sent Events streaming and per-session conversation memory. The same pipeline is available as a Python API and a one-shot CLI query when you don't need a server.

It's provider-light by design: LLM calls go through [OpenRouter](https://openrouter.ai/), so you can swap between models (DeepSeek, Claude, GPT, and others) by changing one env var, with automatic fallback to backup models when one is unavailable.

## Why KSS RAG

Most RAG frameworks are libraries — powerful, but they hand you primitives and expect you to assemble the server, the streaming, and the session handling yourself. KSS RAG ships that assembly as a first-class feature:

- **One command to a running API.** `kssrag server --file docs.pdf --system-prompt prompt.txt` gives you `/query`, `/stream` (SSE), `/health`, and session management out of the box.
- **Bring your own prompt and source of truth.** The system prompt and the document are inputs, not code changes.
- **Pluggable everything.** Six vector stores, two retrievers, multiple chunkers — selected by config, or replaced entirely with your own classes via an import path (no forking required).
- **Streaming that doesn't leak internals.** Token-by-token SSE with marker-aware buffering (see [Conversation memory](#conversation-memory)).
- **Rolling conversation memory.** The agent compresses history into running summaries to keep context bounded across long conversations.

If you want a RAG *service* rather than a RAG *toolkit*, that's the niche this fills.

## Quick Start

### Install

```bash
pip install kssrag

# Optional extras
pip install kssrag[ocr]      # PaddleOCR (handwritten) + Tesseract (typed)
pip install kssrag[office]   # DOCX / Excel / PowerPoint loaders
pip install kssrag[gpu]      # GPU FAISS
pip install kssrag[all]      # everything
```

Set your key (see `.env.example` for all options):

```bash
echo "OPENROUTER_API_KEY=your_key_here" > .env
```

### Serve a RAG API in one command

```bash
python -m kssrag.cli server \
    --file knowledge_base.txt \
    --system-prompt "You are a support assistant. Answer only from the provided context." \
    --vector-store hybrid_offline \
    --host 0.0.0.0 --port 8000
```

Then query it:

```bash
curl -X POST http://localhost:8000/stream \
    -H "Content-Type: application/json" \
    -d '{"query": "How do I reset my password?", "session_id": "user-123"}'
```

`--system-prompt` accepts either an inline string or a path to a prompt file.

## CLI

Two subcommands: `query` (one-shot) and `server` (persistent API).

```bash
# One-shot query with streaming output
python -m kssrag.cli query \
    --file report.pdf \
    --format pdf \
    --query "Summarize the key risks." \
    --vector-store hybrid_online \
    --top-k 8 \
    --stream

# OCR an image, then query it
python -m kssrag.cli query \
    --file scanned_notes.png \
    --format image \
    --ocr-mode handwritten \
    --query "What are the action items?"
```

| Flag | Applies to | Description |
|------|-----------|-------------|
| `--file` | both | Path to the source document (required) |
| `--query` | query | The question to ask (required) |
| `--format` | both | `text`, `json`, `pdf`, `image`, `docx`, `excel`, `pptx` |
| `--vector-store` | both | `bm25`, `bm25s`, `faiss`, `tfidf`, `hybrid_online`, `hybrid_offline` |
| `--system-prompt` | both | Inline prompt text, or a path to a prompt file |
| `--stream` | query | Stream the response token-by-token |
| `--top-k` | query | Number of chunks to retrieve |
| `--ocr-mode` | query | `typed` (Tesseract) or `handwritten` (PaddleOCR) |
| `--host` / `--port` | server | Server bind address |

> **Note:** the `server` subcommand currently loads `text`, `json`, and `pdf` formats. Image and Office formats are supported by the `query` subcommand.

## Python API

```python
from kssrag import KSSRAG, Config, VectorStoreType

config = Config(
    OPENROUTER_API_KEY="your-key",
    VECTOR_STORE_TYPE=VectorStoreType.HYBRID_OFFLINE,
    CHUNK_SIZE=800,
    TOP_K=8,
)

rag = KSSRAG(config=config)
rag.load_document("technical_docs.pdf")

# Blocking query
print(rag.query("What are the technical specifications?"))

# Streaming query
for chunk in rag.agent.query_stream("Walk me through the architecture.", top_k=8):
    print(chunk, end="", flush=True)
```

> **Note:** the `KSSRAG` class auto-detects `.txt`, `.json`, and `.pdf`. For images and Office documents, use the CLI or construct the matching chunker (`ImageChunker`, `OfficeChunker`) directly.

### Custom components

Any pipeline stage can be replaced with your own class — no fork needed. Point config at an import path:

```python
config = Config(
    CUSTOM_VECTOR_STORE="my_module.MyVectorStore",
    CUSTOM_RETRIEVER="my_module.MyRetriever",
    CUSTOM_LLM="my_module.MyLLM",
)
```

Custom vector stores subclass `BaseVectorStore` (`add_documents` / `retrieve` / `persist` / `load`); retrievers subclass `BaseRetriever` (`retrieve`).

## Serve as an embedded API

```python
from kssrag import KSSRAG
import uvicorn

rag = KSSRAG()
rag.load_document("knowledge.txt")
app, server_config = rag.create_server()
uvicorn.run(app, host="0.0.0.0", port=8000)
```

### Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/query` | POST | Query the RAG system (`query`, `session_id`) |
| `/stream` | POST | Streaming query via Server-Sent Events |
| `/health` | GET | Health check |
| `/config` | GET | Active server configuration |
| `/sessions/{id}/clear` | GET | Clear a session's conversation history |

Each `session_id` gets its own conversation state (held in memory for the life of the server process). CORS is configurable via environment variables.

## Architecture

The pipeline: **load → chunk → vector store → retriever → agent → LLM**. Each stage is swappable by config or replaceable with a custom class.

```
Document ──> Chunker ──> Vector Store ──> Retriever ──┐
                                                       ├──> RAG Agent ──> OpenRouter LLM ──> Response (stream / blocking)
                              Query ───────────────────┘
```

### Vector stores

| Store | Method | Needs model download? | Best for |
|-------|--------|:---:|----------|
| `bm25` | Keyword (BM25Okapi) | No | Fast keyword search |
| `bm25s` | Stemmed BM25 (bm25s lib) | No | Faster BM25 with stemming |
| `tfidf` | TF-IDF + cosine | No | Statistical relevance |
| `faiss` | Dense embeddings (SentenceTransformers) | Yes | Semantic search |
| `hybrid_online` | BM25 + FAISS, embedding-reranked | Yes | Best semantic quality |
| `hybrid_offline` | BM25 + TF-IDF, score-fused | **No** | Semantic-ish quality, zero downloads, air-gapped |

`hybrid_offline` is the default — it needs no network access or model download, which makes it a solid choice for restricted environments.

### Chunkers

`TextChunker` (character windows with overlap) is the base. `PDFChunker`, `ImageChunker` (OCR), and `OfficeChunker` extract text and delegate to it; `JSONChunker` flattens records keyed on a `name` field. Every chunk is a `{"content", "metadata"}` dict carried through the whole pipeline.

### Conversation memory

To keep long conversations from blowing up context windows, the agent maintains **rolling summaries**: after a couple of exchanges it asks the model to append a compact `[SUMMARY_START]...[SUMMARY_END]` block to each response, extracts and stores it, and strips it before the user ever sees it. Streaming is marker-aware — it buffers around partial markers at chunk boundaries so a summary can never leak mid-stream. Older raw turns are trimmed while their summaries are retained, so the agent "remembers" the gist without paying for the full transcript.

### FAISS loading

FAISS is imported lazily and only when a FAISS-backed store is actually used. It probes AVX512 → AVX2 → standard builds in order, so it runs on machines without AVX2 (including many Windows setups) instead of hard-failing at import.

## Configuration

Everything is configurable through environment variables (`.env`) or the `Config` object. Highlights:

```bash
OPENROUTER_API_KEY=your_key
DEFAULT_MODEL=deepseek/deepseek-chat-v3.1:free
FALLBACK_MODELS=deepseek/deepseek-r1:free,deepseek/deepseek-chat

CHUNK_SIZE=500
CHUNK_OVERLAP=50
VECTOR_STORE_TYPE=hybrid_offline
RETRIEVER_TYPE=simple
TOP_K=5

SERVER_HOST=localhost
SERVER_PORT=8000
CORS_ORIGINS=*
```

See `.env.example` for the complete list, including OCR mode, batch size, fuzzy-match threshold, CORS details, and custom-component import paths.

## Development

```bash
git clone https://github.com/Ksschkw/kssrag
cd kssrag
pip install -e .[dev,ocr,all]

python -m pytest tests/ -v        # run tests
python -m pytest tests/test_basic.py::test_text_rag -v   # single test

black kssrag/ tests/             # format
flake8 kssrag/                   # lint
mypy kssrag/                     # type-check
```

## Acknowledgments

Built on [FAISS](https://github.com/facebookresearch/faiss), [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR), [SentenceTransformers](https://github.com/UKPLab/sentence-transformers), [rank-bm25](https://github.com/dorianbrown/rank_bm25) / [bm25s](https://github.com/xhluca/bm25s), and [OpenRouter](https://openrouter.ai/).

## Links

- [PyPI](https://pypi.org/project/kssrag/)
- [Issues](https://github.com/Ksschkw/kssrag/issues)
- [Documentation](docs/)

## License

MIT — see [LICENSE](LICENSE).
