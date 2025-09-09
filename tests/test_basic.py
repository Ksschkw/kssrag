import pytest
import os
import tempfile
from kssrag import KSSRAG, Config

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
            MAX_DOCS_FOR_TESTING=1
        )
        
        rag = KSSRAG(config=config)
        rag.load_document(temp_file, format="text")
        
        # Test query
        response = rag.query("What is this document about?")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
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