"""Metadata filtering tests: stores, retrievers, and the RAG agent.

`metadata_filter` restricts retrieval candidates to documents whose metadata
matches every key/value pair (AND semantics).
"""
import pytest

from kssrag.core.vectorstores import (
    BM25VectorStore,
    TFIDFVectorStore,
    BM25SVectorStore,
)
from kssrag.core.retrievers import SimpleRetriever, HybridRetriever

DOCS = [
    {"content": "python data science guide", "metadata": {"source": "docs", "lang": "en"}},
    {"content": "python robotics notes", "metadata": {"source": "notes", "lang": "en"}},
    {"content": "python data science guide FR", "metadata": {"source": "docs", "lang": "fr"}},
]


def _store(cls):
    store = cls()
    store.add_documents(DOCS)
    return store


@pytest.mark.parametrize("cls", [BM25VectorStore, TFIDFVectorStore, BM25SVectorStore])
def test_filter_narrows_results(cls):
    store = _store(cls)
    assert len(store.retrieve("python", top_k=10)) == 3
    docs_res = store.retrieve("python", top_k=10, metadata_filter={"source": "docs"})
    assert all(d["metadata"]["source"] == "docs" for d in docs_res)
    assert len(docs_res) == 2
    fr_res = store.retrieve("python", top_k=10, metadata_filter={"lang": "fr"})
    assert [d["metadata"]["lang"] for d in fr_res] == ["fr"]


@pytest.mark.parametrize("cls", [BM25VectorStore, TFIDFVectorStore, BM25SVectorStore])
def test_filter_no_match_returns_empty(cls):
    store = _store(cls)
    assert store.retrieve("python", top_k=10, metadata_filter={"source": "nope"}) == []


def test_filter_uses_and_semantics():
    store = _store(BM25VectorStore)
    res = store.retrieve("python", top_k=10, metadata_filter={"source": "docs", "lang": "fr"})
    assert len(res) == 1
    assert res[0]["metadata"]["lang"] == "fr"


def test_simple_retriever_passes_filter_through():
    store = _store(BM25VectorStore)
    res = SimpleRetriever(store).retrieve(
        "python", top_k=10, metadata_filter={"source": "notes"}
    )
    assert [d["metadata"]["source"] for d in res] == ["notes"]


def test_hybrid_retriever_passes_filter_through():
    store = _store(BM25VectorStore)
    res = HybridRetriever(store, entity_names=["python"]).retrieve(
        "python", top_k=10, metadata_filter={"source": "docs"}
    )
    assert all(d["metadata"]["source"] == "docs" for d in res)


def test_agent_query_uses_filter_for_context():
    from kssrag.core.agents import RAGAgent

    class _LLM:
        def __init__(self):
            self.calls = []

        def predict(self, messages):
            self.calls.append(messages)
            return "ok"

    llm = _LLM()
    agent = RAGAgent(SimpleRetriever(_store(BM25VectorStore)), llm)
    agent.query("python", metadata_filter={"source": "notes"})
    user_msg = [m for m in llm.calls[0] if m["role"] == "user"][-1]["content"]
    assert "robotics" in user_msg
    assert "data science" not in user_msg
