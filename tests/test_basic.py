import pytest
import os
import tempfile
from kssrag import KSSRAG, Config
from kssrag.models.openrouter import LLMError

def test_text_rag():
    """Test basic text RAG functionality"""
    # Create a temporary text file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("This is a test document about artificial intelligence and machine learning.")
        temp_file = f.name

    try:
        # Initialize with test config
        config = Config(
            OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY", "test_key"),
            MAX_DOCS_FOR_TESTING=1,
            ENABLE_CACHE=False,
        )

        rag = KSSRAG(config=config)
        rag.load_document(temp_file, format="text")

        # Retrieval is deterministic and offline — assert on it directly.
        docs = rag.retriever.retrieve("What is this document about?", top_k=1)
        assert len(docs) > 0
        assert "machine learning" in docs[0]["content"]

        # The LLM call needs a valid API key; when absent it raises LLMError.
        try:
            response = rag.query("What is this document about?")
            assert isinstance(response, str)
            assert len(response) > 0
        except LLMError:
            pytest.skip("No usable OPENROUTER_API_KEY; skipped live LLM assertion")

    finally:
        # Clean up
        os.unlink(temp_file)

def test_config_validation():
    """Test configuration validation"""
    config = Config(
        OPENROUTER_API_KEY="test_key",
        CHUNK_SIZE=500,
        CHUNK_OVERLAP=50
    )
    
    assert config.OPENROUTER_API_KEY == "test_key"
    assert config.CHUNK_SIZE == 500
    assert config.CHUNK_OVERLAP == 50