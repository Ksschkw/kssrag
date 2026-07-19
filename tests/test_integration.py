import pytest
import tempfile
import os
from kssrag import KSSRAG, Config
from kssrag.models.openrouter import LLMError

def test_bm25s_integration():
    """Test BM25S integration with KSSRAG"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
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
        assert len(docs) > 0
        assert "Python" in docs[0]["content"]

        # The LLM call needs a valid API key; when absent it raises LLMError.
        try:
            response = rag.query("Python programming")
            assert isinstance(response, str) and len(response) > 0
        except LLMError:
            pytest.skip("No usable OPENROUTER_API_KEY; skipped live LLM assertion")

    finally:
        os.unlink(temp_file)

def test_streaming_integration():
    """Test streaming integration (mock test)"""
    config = Config(ENABLE_STREAMING=True)

    # This is a basic test that config is accepted
    # Actual streaming would require API calls
    assert config.ENABLE_STREAMING == True