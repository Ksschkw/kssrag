"""
Integration tests for KSS RAG components.
Conforms to house-standard.md §15 (no useless or always-passing tests).
"""
import os
import tempfile
import pytest

from kssrag import KSSRAG, Config
from kssrag.core.agents import RAGAgent
from kssrag.models.base import BaseLLM, LLMError


class _DeterministicIntegrationLLM(BaseLLM):
    def predict(self, messages):
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return f"Integrated Answer for {user_msg}"

    def predict_stream(self, messages):
        yield "Integrated "
        yield "Answer "
        yield "Stream"


def test_bm25s_integration():
    """Test BM25S integration with KSSRAG end-to-end."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Test document about Python programming and machine learning.")
        temp_file = f.name

    try:
        config = Config(
            VECTOR_STORE_TYPE="bm25s",
            MAX_DOCS_FOR_TESTING=1,
            ENABLE_CACHE=False,
        )

        rag = KSSRAG(config=config)
        rag.load_document(temp_file, format="text")

        # Retrieval is deterministic and offline — assert on it directly.
        docs = rag.retriever.retrieve("Python programming", top_k=1)
        assert len(docs) == 1
        assert "Python" in docs[0]["content"]

        # If live API key is missing, inject deterministic LLM to test full pipeline without skipping
        if rag.agent.llm is None:
            rag.agent = RAGAgent(rag.retriever, _DeterministicIntegrationLLM())

        response = rag.query("Python programming")
        assert isinstance(response, str)
        assert len(response) > 0
        assert "Integrated Answer" in response or len(response) > 10

    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)


def test_streaming_integration():
    """Test real streaming token assembly through KSSRAG agent (§15)."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("Streaming test document content.")
        temp_file = f.name

    try:
        config = Config(
            VECTOR_STORE_TYPE="bm25",
            ENABLE_CACHE=False,
            ENABLE_STREAMING=True,
        )
        rag = KSSRAG(config=config)
        rag.load_document(temp_file, format="text")

        # Inject streaming adapter to verify stream token pipeline
        rag.agent = RAGAgent(rag.retriever, _DeterministicIntegrationLLM())

        tokens = list(rag.agent.query_stream("test query"))
        assert len(tokens) == 3
        assert "".join(tokens) == "Integrated Answer Stream"
        # Assert assistant response was recorded in conversation
        assert any(m["role"] == "assistant" for m in rag.agent.conversation)
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)
