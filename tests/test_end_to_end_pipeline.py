"""
Comprehensive end-to-end integration tests for KSSRAG pipeline.
Tests document loading, chunking, indexing, persistence, retrieval, stealth summarization,
and question answering across vector store types with zero live network calls.
Fails if any small contract, structure, or data type is altered.
"""
import json
import os
import tempfile
import pytest

from kssrag import KSSRAG, Config
from kssrag.config import VectorStoreType, ChunkerType, RetrieverType
from kssrag.core.agents import RAGAgent
from kssrag.core.retrievers import SimpleRetriever, HybridRetriever
from kssrag.core.vectorstores import BM25VectorStore, BM25SVectorStore, TFIDFVectorStore, HybridOfflineVectorStore
from kssrag.models.base import BaseLLM


class DeterministicMockLLM(BaseLLM):
    """Predictable mock LLM for deterministic integration assertions."""
    def __init__(self, answer_prefix: str = "ANSWER: "):
        self.answer_prefix = answer_prefix
        self.call_history = []

    def predict(self, messages):
        self.call_history.append(messages)
        user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        if "Question: " in user_msg:
            question = user_msg.split("Question: ", 1)[1].strip()
        else:
            question = user_msg.strip()

        # Check if stealth summary instruction was included by the agent
        has_summary_instruction = any(
            "[SUMMARY_START]" in m.get("content", "") for m in messages if m.get("role") == "system"
        )
        if has_summary_instruction:
            return f"{self.answer_prefix}{question}\n\n[SUMMARY_START]Key context: {question}[SUMMARY_END]"
        return f"{self.answer_prefix}{question}"

    def predict_stream(self, messages):
        full = self.predict(messages)
        # Yield in multi-token chunks to test partial delimiter boundaries
        parts = [full[:10], full[10:25], full[25:]]
        for part in parts:
            if part:
                yield part


def test_e2e_text_pipeline_with_bm25_offline():
    """Verify entire pipeline with plain text file and BM25 offline vector store."""
    content = (
        "Antigravity is an advanced agentic coding framework.\n"
        "It emphasizes strict layered architecture and high test fidelity.\n"
        "Circuit breakers protect outbound API boundaries from cascading failure."
    )
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(content)
        file_path = f.name

    try:
        config = Config(
            VECTOR_STORE_TYPE=VectorStoreType.BM25,
            CHUNKER_TYPE=ChunkerType.TEXT,
            CHUNK_SIZE=120,
            CHUNK_OVERLAP=20,
            ENABLE_CACHE=False,
        )
        rag = KSSRAG(config=config)
        rag.load_document(file_path, format="text")

        # Verify documents were chunked properly
        assert len(rag.documents) >= 2
        for doc in rag.documents:
            assert "content" in doc
            assert "metadata" in doc
            assert doc["metadata"]["source"] == file_path

        # Inject deterministic LLM
        mock_llm = DeterministicMockLLM()
        rag.agent = RAGAgent(rag.retriever, mock_llm)

        # Execute query
        answer = rag.query("What protects outbound API boundaries?", top_k=2)
        assert "ANSWER:" in answer
        assert len(mock_llm.call_history) == 1
        last_call = mock_llm.call_history[0]

        # Verify context injection contains the retrieved document
        prompt_content = " ".join(m["content"] for m in last_call)
        assert "Circuit breakers protect outbound" in prompt_content
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


def test_e2e_json_pipeline_with_hybrid_retriever():
    """Verify JSON ingestion with entity-boosting HybridRetriever."""
    data = [
        {"name": "Resilience Module", "description": "Implements circuit breaker, timeout, and jittered retries."},
        {"name": "Security Module", "description": "Provides input sanitization and security header middleware."},
        {"name": "Logging Module", "description": "Provides structured logging with secret redaction rules."},
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        file_path = f.name

    try:
        config = Config(
            VECTOR_STORE_TYPE=VectorStoreType.HYBRID_OFFLINE,
            RETRIEVER_TYPE=RetrieverType.HYBRID,
            ENABLE_CACHE=False,
        )
        rag = KSSRAG(config=config)
        rag.load_document(file_path, format="json")

        assert isinstance(rag.retriever, HybridRetriever)
        assert "Resilience Module" in rag.retriever.entity_names

        # Retrieval query matching exact entity name should rank first
        results = rag.retriever.retrieve("Resilience Module", top_k=1)
        assert len(results) == 1
        assert "circuit breaker" in results[0]["content"]

        mock_llm = DeterministicMockLLM()
        rag.agent = RAGAgent(rag.retriever, mock_llm)
        response = rag.query("Tell me about Resilience Module", top_k=1)
        assert "ANSWER: Tell me about Resilience Module" in response
    finally:
        if os.path.exists(file_path):
            os.unlink(file_path)


def test_e2e_stealth_summarization_integration():
    """Verify stealth summary blocks are parsed, stored, and hidden from the user."""
    mock_llm = DeterministicMockLLM()
    store = BM25VectorStore()
    store.add_documents([
        {"content": "Session state is preserved in memory.", "metadata": {"id": 1}},
    ])
    retriever = SimpleRetriever(store)
    agent = RAGAgent(retriever, mock_llm)

    # In kssrag, stealth summarization instructions activate once exchange_count >= 2.
    # Because only assistant messages are saved to conversation history,
    # exchange_count = len(assistant_replies) // 2, which requires 4 prior turns.
    for i in range(1, 5):
        resp = agent.query(f"Question {i}")
        assert resp == f"ANSWER: Question {i}"
        assert len(agent.conversation_summaries) == 0

    # Turn 5: exchange_count >= 2, stealth summary instruction is active
    resp5 = agent.query("How is state preserved?")
    # The summary tag MUST be stripped before returning to user
    assert "[SUMMARY_START]" not in resp5
    assert "[SUMMARY_END]" not in resp5
    assert resp5 == "ANSWER: How is state preserved?"
    # The summary MUST be preserved in agent internal state
    assert len(agent.conversation_summaries) == 1
    assert "Key context: How is state preserved?" in agent.conversation_summaries[0]


def test_e2e_streaming_stealth_summary_protection():
    """Verify that streaming output terminates before [SUMMARY_START] reaches client."""
    mock_llm = DeterministicMockLLM()
    store = BM25VectorStore()
    store.add_documents([{"content": "Streaming documentation.", "metadata": {"id": 1}}])
    retriever = SimpleRetriever(store)
    agent = RAGAgent(retriever, mock_llm)

    # 4 prior turns to reach exchange_count >= 2
    for i in range(1, 5):
        agent.query(f"Setup turn {i}")

    # Streaming turn with active summary instructions
    stream_chunks = list(agent.query_stream("Tell me more"))
    streamed_text = "".join(stream_chunks)

    # Asserts that no summary leaked in the stream
    assert "[SUMMARY_START]" not in streamed_text
    assert "[SUMMARY_END]" not in streamed_text
    assert "Key context" not in streamed_text
    assert streamed_text.strip() == "ANSWER: Tell me more"


def test_e2e_cache_persistence_and_reload():
    """Verify index cache creation, persistence to disk, and instant reload."""
    with tempfile.TemporaryDirectory() as cache_dir:
        content = "Cache persistence test document."
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
            f.write(content)
            source_file = f.name

        try:
            config = Config(
                VECTOR_STORE_TYPE=VectorStoreType.BM25,
                ENABLE_CACHE=True,
                CACHE_DIR=cache_dir,
            )

            # First run: build and persist
            rag1 = KSSRAG(config=config)
            rag1.load_document(source_file, format="text")
            docs1 = rag1.retriever.retrieve("cache", top_k=1)
            assert len(docs1) == 1

            # Second run: reload from cache
            rag2 = KSSRAG(config=config)
            rag2.load_document(source_file, format="text")
            docs2 = rag2.retriever.retrieve("cache", top_k=1)
            assert len(docs2) == 1
            assert docs1[0]["content"] == docs2[0]["content"]
        finally:
            if os.path.exists(source_file):
                os.unlink(source_file)
