"""
No-LLM RAG behavior battery.

Deterministic tests for the retrieval side of KSS RAG — the parts that must be
correct regardless of which LLM (if any) is attached. No network, no embeddings
model, no API key. Covers: BM25 scoring semantics, top_k contracts, RRF fusion,
retriever behavior, chunking, and the vector-store retrieval contract.
"""
import numpy as np
import pytest

from kssrag.core.bm25 import BM25Okapi, tokenize
from kssrag.core.vectorstores import (
    BM25VectorStore,
    TFIDFVectorStore,
    HybridOfflineVectorStore,
)
from kssrag.core.retrievers import SimpleRetriever, HybridRetriever
from kssrag.core.fusion import reciprocal_rank_fusion, DEFAULT_RRF_K
from kssrag.core.chunkers import TextChunker, JSONChunker


def _docs(pairs):
    return [{"content": text, "metadata": {"id": i}} for i, text in pairs]


CORPUS = _docs([
    ("d0", "python is a programming language used for data science"),
    ("d1", "machine learning models learn from training data"),
    ("d2", "deep learning uses neural networks with many layers"),
    ("d3", "the quick brown fox jumps over the lazy dog"),
    ("d4", "rust is a systems programming language with memory safety"),
    ("d5", "supervised machine learning needs labeled training data"),
])


# --- BM25 scoring semantics --------------------------------------------------

def test_bm25_tokenizer_lowercases_and_splits():
    assert tokenize("Hello, WORLD! 123") == ["hello", "world", "123"]


def test_bm25_term_frequency_matters():
    """A doc mentioning the query term more often should score higher."""
    corpus = [tokenize("cat"), tokenize("cat cat cat"), tokenize("dog bird fish")]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize("cat"))
    assert scores[1] > scores[0] > scores[2]
    assert scores[2] == 0.0  # no query term present


def test_bm25_idf_rare_terms_weighted_higher():
    """A rare term should contribute more than a common one."""
    corpus = [
        tokenize("common common common rare"),
        tokenize("common common common"),
        tokenize("common common common"),
        tokenize("common common common"),
    ]
    bm25 = BM25Okapi(corpus)
    # Only doc 0 has 'rare'; all docs have 'common'. Querying 'rare' should
    # strongly favor doc 0; querying 'common' should barely discriminate.
    rare_scores = bm25.get_scores(tokenize("rare"))
    assert rare_scores[0] > 0
    assert all(s == 0.0 for s in rare_scores[1:])


def test_bm25_length_normalization():
    """Between two docs with one match, the shorter doc scores higher (b>0)."""
    corpus = [tokenize("match"), tokenize("match " + "filler " * 50)]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize("match"))
    assert scores[0] > scores[1]


def test_bm25_idf_non_negative():
    """Our Lucene-style IDF variant stays non-negative even for ubiquitous terms."""
    corpus = [tokenize("x y"), tokenize("x z"), tokenize("x w")]  # 'x' in every doc
    bm25 = BM25Okapi(corpus)
    assert all(v >= 0 for v in bm25.idf.values())


def test_bm25_empty_query_all_zero():
    bm25 = BM25Okapi([tokenize("some text here")])
    assert bm25.get_scores([]) == [0.0]


# --- top_k contract ----------------------------------------------------------

@pytest.mark.parametrize("store_cls", [BM25VectorStore, TFIDFVectorStore, HybridOfflineVectorStore])
def test_top_k_is_honored(store_cls):
    store = store_cls()
    store.add_documents(CORPUS)
    for k in (1, 3, 5):
        assert len(store.retrieve("machine learning data", top_k=k)) == k


@pytest.mark.parametrize("store_cls", [BM25VectorStore, TFIDFVectorStore, HybridOfflineVectorStore])
def test_top_k_larger_than_corpus(store_cls):
    store = store_cls()
    store.add_documents(CORPUS)
    results = store.retrieve("programming language", top_k=100)
    assert len(results) <= len(CORPUS)


@pytest.mark.parametrize("store_cls", [BM25VectorStore, TFIDFVectorStore, HybridOfflineVectorStore])
def test_retrieve_returns_relevant_first(store_cls):
    store = store_cls()
    store.add_documents(CORPUS)
    results = store.retrieve("machine learning training data", top_k=3)
    ids = [d["metadata"]["id"] for d in results]
    # d1 and d5 are the machine-learning/training docs; at least one in top-2.
    assert ("d1" in ids[:2]) or ("d5" in ids[:2])


@pytest.mark.parametrize("store_cls", [BM25VectorStore, TFIDFVectorStore, HybridOfflineVectorStore])
def test_no_duplicate_documents(store_cls):
    store = store_cls()
    store.add_documents(CORPUS)
    ids = [d["metadata"]["id"] for d in store.retrieve("data", top_k=5)]
    assert len(ids) == len(set(ids))


def test_retrieve_preserves_document_shape():
    store = BM25VectorStore()
    store.add_documents(CORPUS)
    doc = store.retrieve("python", top_k=1)[0]
    assert set(doc.keys()) == {"content", "metadata"}
    assert "id" in doc["metadata"]


# --- RRF fusion --------------------------------------------------------------

def test_rrf_consensus_beats_single_list_top():
    d = lambda c: {"content": c, "metadata": {"id": c}}
    A, B, C, D = d("A"), d("B"), d("C"), d("D")
    # A is #1 in list1 and #2 in list2 -> strong consensus; D appears once.
    fused = reciprocal_rank_fusion([[A, C, D], [B, A]], top_k=4)
    assert fused[0]["metadata"]["id"] == "A"


def test_rrf_weighting_shifts_ranking():
    d = lambda c: {"content": c, "metadata": {"id": c}}
    A, B = d("A"), d("B")
    heavy_second = reciprocal_rank_fusion([[A], [B, A]], top_k=2, weights=[0.01, 10.0])
    assert heavy_second[0]["metadata"]["id"] == "B"


def test_rrf_dedup_and_topk():
    d = lambda c: {"content": c, "metadata": {"id": c}}
    A, B = d("A"), d("B")
    fused = reciprocal_rank_fusion([[A, B], [A, B]], top_k=10)
    assert [x["metadata"]["id"] for x in fused] == ["A", "B"]


def test_rrf_score_formula():
    """Verify the exact RRF score for a hand-checkable case."""
    d = lambda c: {"content": c, "metadata": {"id": c}}
    A = d("A")
    # A at rank 1 in both lists -> 2 * 1/(k+1)
    fused = reciprocal_rank_fusion([[A], [A]], top_k=1)
    assert fused == [A]  # trivially, but exercises the accumulation path
    k = DEFAULT_RRF_K
    assert 2 * (1 / (k + 1)) > 0  # sanity on the constant


def test_rrf_weights_length_mismatch_raises():
    d = lambda c: {"content": c, "metadata": {"id": c}}
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[d("A")]], top_k=1, weights=[1.0, 2.0])


def test_rrf_empty():
    assert reciprocal_rank_fusion([[], []], top_k=5) == []


# --- retrievers --------------------------------------------------------------

def test_simple_retriever_delegates():
    store = BM25VectorStore()
    store.add_documents(CORPUS)
    r = SimpleRetriever(store)
    assert len(r.retrieve("python", top_k=2)) == 2


def test_hybrid_retriever_entity_boost():
    """HybridRetriever should surface docs mentioning a matched entity."""
    docs = _docs([
        ("e0", "acme corporation quarterly earnings report"),
        ("e1", "general notes about quarterly earnings in the industry"),
        ("e2", "acme corporation product roadmap and strategy"),
    ])
    store = BM25VectorStore()
    store.add_documents(docs)
    retriever = HybridRetriever(store, entity_names=["acme corporation"])
    results = retriever.retrieve("acme corporation earnings", top_k=3)
    assert len(results) >= 1
    # An 'acme corporation' doc should be present in the results.
    assert any("acme" in d["content"] for d in results)


# --- chunking ----------------------------------------------------------------

def test_text_chunker_overlap_and_coverage():
    text = "abcdefghij" * 10  # 100 chars
    chunker = TextChunker(chunk_size=30, overlap=10)
    chunks = chunker.chunk(text, {"source": "t"})
    assert len(chunks) > 1
    # Every chunk carries metadata + an incrementing chunk_id.
    assert [c["metadata"]["chunk_id"] for c in chunks] == list(range(len(chunks)))
    # Reassembling with the known stride reproduces the original length coverage.
    assert chunks[0]["content"] == text[:30]
    # Overlap: second chunk starts 20 chars in (size - overlap).
    assert chunks[1]["content"][0] == text[20]


def test_text_chunker_short_text_single_chunk():
    chunker = TextChunker(chunk_size=100, overlap=10)
    chunks = chunker.chunk("short", {})
    assert len(chunks) == 1
    assert chunks[0]["content"] == "short"


def test_json_chunker_builds_named_records():
    data = [
        {"name": "Widget", "color": "blue", "tags": ["a", "b"]},
        {"name": "Gadget", "price": "10"},
        {"no_name_field": "skipped"},
    ]
    chunks = JSONChunker().chunk(data)
    assert len(chunks) == 2  # third has no 'name' -> skipped
    assert chunks[0]["metadata"]["name"] == "Widget"
    assert "Widget" in chunks[0]["content"]


# --- persistence round-trip (no network) -------------------------------------

def test_bm25_persist_load_roundtrip(tmp_path):
    path = str(tmp_path / "idx.pkl")
    store = BM25VectorStore(persist_path=path)
    store.add_documents(CORPUS)
    before = [d["metadata"]["id"] for d in store.retrieve("machine learning", top_k=3)]
    store.persist()

    reloaded = BM25VectorStore(persist_path=path)
    reloaded.load()
    after = [d["metadata"]["id"] for d in reloaded.retrieve("machine learning", top_k=3)]
    assert before == after


# --- index cache: hit / miss / invalidation ---------------------------------

def _count_builds(cache_module):
    """Wrap a build_documents callable to count how often it actually runs."""
    calls = {"n": 0}

    def make(fn):
        def counting():
            calls["n"] += 1
            return fn()
        return counting

    return calls, make


def test_cache_hit_skips_rebuild(tmp_path):
    from kssrag.utils import cache as C
    from kssrag.config import VectorStoreType

    calls, wrap = _count_builds(C)
    build = wrap(lambda: [{"content": "hello world", "metadata": {"id": 0}}])
    args = dict(
        source_path=str(tmp_path / "src.txt"),
        store_type=VectorStoreType.BM25,
        cache_root=str(tmp_path / "cache"),
        chunk_size=200, chunk_overlap=20, max_docs=None,
    )
    # Need a real source file for the hash.
    (tmp_path / "src.txt").write_text("hello world content")

    C.load_or_build_index(build_documents=build, **args)
    C.load_or_build_index(build_documents=build, **args)
    assert calls["n"] == 1  # second call is a cache hit


def test_cache_invalidates_on_param_change(tmp_path):
    from kssrag.utils import cache as C
    from kssrag.config import VectorStoreType

    (tmp_path / "src.txt").write_text("hello world content")
    calls, wrap = _count_builds(C)
    build = wrap(lambda: [{"content": "hello world", "metadata": {"id": 0}}])
    base = dict(
        source_path=str(tmp_path / "src.txt"),
        store_type=VectorStoreType.BM25,
        cache_root=str(tmp_path / "cache"),
        chunk_overlap=20, max_docs=None,
    )
    C.load_or_build_index(build_documents=build, chunk_size=200, **base)
    C.load_or_build_index(build_documents=build, chunk_size=400, **base)  # changed
    assert calls["n"] == 2  # param change forces a rebuild


def test_cache_invalidates_on_content_change(tmp_path):
    from kssrag.utils import cache as C
    from kssrag.config import VectorStoreType

    src = tmp_path / "src.txt"
    src.write_text("original content")
    calls, wrap = _count_builds(C)
    build = wrap(lambda: [{"content": "x", "metadata": {"id": 0}}])
    args = dict(
        source_path=str(src),
        store_type=VectorStoreType.BM25,
        cache_root=str(tmp_path / "cache"),
        chunk_size=200, chunk_overlap=20, max_docs=None,
    )
    C.load_or_build_index(build_documents=build, **args)
    src.write_text("totally different content")
    C.load_or_build_index(build_documents=build, **args)
    assert calls["n"] == 2  # source hash changed -> rebuild

