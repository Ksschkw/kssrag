"""
Index persistence / caching for KSS RAG.

Keyed on the source file path so that re-indexing the same document reuses a
single cache directory instead of accumulating copies. A plain-JSON manifest
records the source content hash, the vector-store type, and the chunk
parameters; the manifest is checked BEFORE any deserialization so that cache
invalidation never requires loading a pickle.

Cache files are locally generated and trusted-local-only. They contain pickled
objects (sklearn/bm25 fitted state) and should not be shared between machines
or users.
"""
import hashlib
import json
import os
import shutil
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..config import VectorStoreType
from .helpers import logger

MANIFEST_NAME = "manifest.json"
DOCUMENTS_NAME = "documents.json"
INDEX_BASENAME = "index"


def compute_source_hash(file_path: str) -> str:
    """SHA-256 of a source file's bytes, read in chunks to bound memory."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def cache_dir_for(source_path: str, cache_root: str) -> str:
    """Stable per-source cache directory (keyed on absolute path)."""
    key = hashlib.sha256(os.path.abspath(source_path).encode("utf-8")).hexdigest()[:16]
    return os.path.join(cache_root, key)


def create_vector_store(store_type: str, persist_base: str):
    """Construct a vector store whose persistence paths live under persist_base."""
    from ..core.vectorstores import (
        BM25VectorStore,
        BM25SVectorStore,
        FAISSVectorStore,
        TFIDFVectorStore,
        HybridVectorStore,
        HybridOfflineVectorStore,
    )

    if store_type == VectorStoreType.BM25:
        return BM25VectorStore(persist_base + ".bm25.pkl")
    if store_type == VectorStoreType.BM25S:
        return BM25SVectorStore(persist_base + ".bm25s.pkl")
    if store_type == VectorStoreType.TFIDF:
        return TFIDFVectorStore(persist_base + ".tfidf.pkl")
    if store_type == VectorStoreType.FAISS:
        return FAISSVectorStore(persist_base + ".faiss")
    if store_type == VectorStoreType.HYBRID_ONLINE:
        return HybridVectorStore(persist_base + ".hybrid")
    if store_type == VectorStoreType.HYBRID_OFFLINE:
        return HybridOfflineVectorStore(persist_base + ".hybrid_offline")
    raise ValueError(f"Unknown vector store type for caching: {store_type}")


def _expected_manifest(source_path: str, store_type: str,
                       chunk_size: int, chunk_overlap: int,
                       max_docs: Optional[int]) -> Dict[str, Any]:
    return {
        "version": 1,
        "source_hash": compute_source_hash(source_path),
        "store_type": str(store_type),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "max_docs": max_docs,
    }


def load_or_build_index(
    source_path: str,
    store_type: str,
    build_documents: Callable[[], List[Dict[str, Any]]],
    cache_root: str,
    chunk_size: int,
    chunk_overlap: int,
    max_docs: Optional[int] = None,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """
    Return (vector_store, documents), reusing a cached index when the source
    content, store type, and chunk parameters all match. Otherwise (re)build
    in place, overwriting any stale cache for this source.

    build_documents() is only invoked on a cache miss, so the expensive
    chunking step is skipped entirely on a hit.
    """
    cache_dir = cache_dir_for(source_path, cache_root)
    manifest_path = os.path.join(cache_dir, MANIFEST_NAME)
    documents_path = os.path.join(cache_dir, DOCUMENTS_NAME)
    index_base = os.path.join(cache_dir, INDEX_BASENAME)

    expected = _expected_manifest(
        source_path, store_type, chunk_size, chunk_overlap, max_docs
    )

    # --- Try cache hit -----------------------------------------------------
    if os.path.exists(manifest_path) and os.path.exists(documents_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest == expected:
                with open(documents_path, "r", encoding="utf-8") as f:
                    documents = json.load(f)
                store = create_vector_store(store_type, index_base)
                store.load()
                if store.documents:
                    logger.info(f"Loaded cached index from {cache_dir}")
                    return store, documents
                logger.info("Cached index empty after load; rebuilding")
            else:
                logger.info("Cache manifest mismatch; rebuilding index")
        except Exception as e:  # noqa: BLE001 - any failure => rebuild
            logger.warning(f"Failed to load cached index ({e}); rebuilding")

    # --- Cache miss: build fresh ------------------------------------------
    documents = build_documents()
    store = create_vector_store(store_type, index_base)
    store.add_documents(documents)

    try:
        # Overwrite any stale cache dir for this source to avoid accumulation.
        if os.path.isdir(cache_dir):
            shutil.rmtree(cache_dir)
        os.makedirs(cache_dir, exist_ok=True)
        store.persist()
        with open(documents_path, "w", encoding="utf-8") as f:
            json.dump(documents, f)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(expected, f)
        logger.info(f"Built and cached index at {cache_dir}")
    except Exception as e:  # noqa: BLE001 - caching is best-effort
        logger.warning(f"Failed to persist index cache ({e}); continuing without cache")

    return store, documents
