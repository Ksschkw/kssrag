"""
FAISS vector store mechanics — without the heavy embedding model.

FAISSVectorStore normally loads a SentenceTransformer (which pulls in torch,
~1GB). These tests inject a tiny deterministic fake embedder so we can exercise
the real FAISS index path — add_documents, retrieve, persist/load — without that
dependency. Skipped entirely if faiss-cpu isn't installed.
"""
import numpy as np
import pytest

faiss = pytest.importorskip("faiss")

from kssrag.core.vectorstores import FAISSVectorStore


class _FakeEmbedder:
    """Deterministic bag-of-chars embedder: no ML, but stable and discriminative."""

    def __init__(self, dim=16):
        self._dim = dim

    def get_sentence_embedding_dimension(self):
        return self._dim

    def encode(self, texts, **kwargs):
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = []
        for t in items:
            v = np.zeros(self._dim, dtype="float32")
            for ch in t.lower():
                v[ord(ch) % self._dim] += 1.0
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            vecs.append(v)
        arr = np.array(vecs, dtype="float32")
        return arr[0] if single else arr


def _patched_init(self, persist_path=None, model_name=None):
    """A FAISS store __init__ that skips setup_faiss and uses the fake embedder."""
    from kssrag.core.vectorstores import BaseVectorStore
    BaseVectorStore.__init__(self, persist_path)
    self.model_name = "fake"
    self.model = _FakeEmbedder()
    self.dimension = self.model.get_sentence_embedding_dimension()
    self.index = faiss.IndexFlatL2(self.dimension)
    self.metadata_path = persist_path + ".meta" if persist_path else None


@pytest.fixture
def faiss_store(monkeypatch):
    monkeypatch.setattr(
        "kssrag.core.vectorstores.FAISSVectorStore.__init__", _patched_init
    )
    return FAISSVectorStore()


DOCS = [
    {"content": "aaaa aaaa aaaa", "metadata": {"id": "a"}},
    {"content": "bbbb bbbb bbbb", "metadata": {"id": "b"}},
    {"content": "cccc cccc cccc", "metadata": {"id": "c"}},
]


def test_faiss_add_and_retrieve(faiss_store):
    faiss_store.add_documents(DOCS)
    assert faiss_store.index.ntotal == 3
    # Query closest to the 'a' doc should return it first.
    results = faiss_store.retrieve("aaaa aaaa", top_k=1)
    assert results[0]["metadata"]["id"] == "a"


def test_faiss_top_k(faiss_store):
    faiss_store.add_documents(DOCS)
    assert len(faiss_store.retrieve("aaaa", top_k=2)) == 2
    assert len(faiss_store.retrieve("aaaa", top_k=99)) == 3


def test_faiss_persist_load(tmp_path, monkeypatch):
    path = str(tmp_path / "faiss.index")
    monkeypatch.setattr(
        "kssrag.core.vectorstores.FAISSVectorStore.__init__", _patched_init
    )
    store = FAISSVectorStore(persist_path=path)
    store.add_documents(DOCS)
    before = [d["metadata"]["id"] for d in store.retrieve("bbbb", top_k=3)]
    store.persist()

    reloaded = FAISSVectorStore(persist_path=path)
    reloaded.load()
    after = [d["metadata"]["id"] for d in reloaded.retrieve("bbbb", top_k=3)]
    assert before == after


def test_hybrid_online_rrf(monkeypatch):
    """HybridVectorStore (BM25 + FAISS, fused via RRF) using the fake embedder."""
    monkeypatch.setattr(
        "kssrag.core.vectorstores.FAISSVectorStore.__init__", _patched_init
    )
    from kssrag.core.vectorstores import HybridVectorStore

    store = HybridVectorStore(persist_path="hybrid_test")
    docs = [
        {"content": "aaaa aaaa programming code", "metadata": {"id": "a"}},
        {"content": "bbbb bbbb database systems", "metadata": {"id": "b"}},
        {"content": "cccc cccc network protocols", "metadata": {"id": "c"}},
    ]
    store.add_documents(docs)
    results = store.retrieve("programming code aaaa", top_k=3)
    ids = [d["metadata"]["id"] for d in results]
    assert len(ids) == len(set(ids))         # no duplicates after fusion
    assert len(ids) <= 3                       # top_k honored
    assert "a" in ids                          # the matching doc is retrieved

