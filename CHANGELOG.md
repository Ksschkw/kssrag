# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-07-20

A large reliability, flexibility, and quality release. Backward compatible with
0.2.x public APIs (`KSSRAG`, `OpenRouterLLM`, the CLI, and the server).

### Added

- **Multi-provider LLM support.** New `OpenAICompatibleLLM` adapter targets any
  OpenAI `/chat/completions` endpoint, plus native adapters for **Anthropic**
  (Messages API) and **Ollama** (`/api/chat`). A provider preset registry covers
  17 providers — openrouter, openai, groq, together, deepseek, fireworks,
  mistral, perplexity, xai, deepinfra, anyscale, anthropic, ollama, and local
  servers (ollama-openai, lmstudio, vllm, llamacpp).
- **`create_llm()` factory** and `PROVIDER` / `LLM_BASE_URL` / `LLM_API_KEY`
  config, selectable via env or CLI flags (`--provider`, `--model`,
  `--base-url`, `--api-key`) on both `query` and `server`.
- **Own BM25 Okapi implementation** (`kssrag.core.bm25`) — dependency-free,
  benchmarked to identical rankings against rank-bm25.
- **Reciprocal Rank Fusion (RRF)** (`kssrag.core.fusion`) for hybrid stores,
  replacing ad-hoc score weighting.
- **Retrieval evaluation harness** (`kssrag.eval`) — recall@k, precision@k, MRR,
  nDCG, plus a runner to compare vector stores on labeled data.
- **Index caching** keyed on source content hash + chunk params, with automatic
  invalidation (honors `ENABLE_CACHE` / `CACHE_DIR`).
- **Bounded LRU session eviction** in the server via `MAX_SESSIONS`.
- **Configurable LLM generation params**: `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`,
  `LLM_TIMEOUT`, `LLM_STREAM_TIMEOUT`.
- Comprehensive test suite (93 passing) covering BM25 semantics, top-k
  contracts, RRF, chunking, caching, server endpoints, and all LLM adapters.

### Changed

- **Server no longer blocks the event loop.** Blocking LLM calls are offloaded to
  a worker thread, so the server handles concurrent requests.
- **LLM failures raise `LLMError`** instead of returning an error string that
  polluted conversation history.
- **Heavy imports are lazy.** `import kssrag` and the download-free stores (bm25,
  bm25s, tfidf, hybrid_offline) no longer pull in faiss/torch.
- **REST semantics**: session clearing is now `POST /sessions/{id}/clear` with a
  new `DELETE /sessions/{id}`; the old mutating `GET` is deprecated but retained.
- `CACHE_DIR` now resolves to a platform-appropriate user cache directory.
- Dropped the `rank-bm25` dependency (replaced by the in-tree implementation).

### Fixed

- Version numbers reconciled across `setup.py`, `__init__.py`, and the server.
- Custom `Config` instances are now honored by the LLM factory (previously the
  global config was read instead).
- `bm25s` vector store worked only on the cached load path; now handled
  everywhere.
- Library log lines are no longer mislabeled as `KSSRAG` (log format uses the
  real logger name).
- OCR tests no longer abort collection of the whole suite when optional OCR
  dependencies are absent.
- Removed broken `paddle_models` submodule gitlinks that shipped as empty dirs.

## [0.2.5] - 2026-05-25

Baseline prior to the 0.3.0 work. OpenRouter-only LLM access; BM25/BM25S/FAISS/
TFIDF/hybrid vector stores; FastAPI server with SSE streaming; CLI; OCR and
Office document support.

[0.3.0]: https://github.com/Ksschkw/kssrag/releases/tag/v0.3.0
[0.2.5]: https://pypi.org/project/kssrag/0.2.5/
