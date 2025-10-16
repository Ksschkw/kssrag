import pytest
import tempfile
import os
from kssrag import KSSRAG, Config

def test_bm25s_integration():
    """Test BM25S integration with KSSRAG"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write("Test document about Python programming and machine learning.")
        temp_file = f.name
    
    try:
        config = Config(
            VECTOR_STORE_TYPE="bm25s",
            MAX_DOCS_FOR_TESTING=1
        )
        
        rag = KSSRAG(config=config)
        rag.load_document(temp_file, format="text")
        
        response = rag.query("Python programming")
        
        assert isinstance(response, str)
        assert len(response) > 0
        
    finally:
        os.unlink(temp_file)

def test_streaming_integration():
    """Test streaming integration (mock test)"""
    config = Config(ENABLE_STREAMING=True)
    
    # This is a basic test that config is accepted
    # Actual streaming would require API calls
    assert config.ENABLE_STREAMING == True