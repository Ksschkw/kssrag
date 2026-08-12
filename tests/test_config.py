import pytest
import os
from kssrag.config import Config, VectorStoreType, ChunkerType

def test_config_new_options():
    """Test new configuration options"""
    config = Config(
        OCR_DEFAULT_MODE="handwritten",
        ENABLE_STREAMING=True,
        VECTOR_STORE_TYPE=VectorStoreType.BM25S
    )
    
    assert config.OCR_DEFAULT_MODE == "handwritten"
    assert config.ENABLE_STREAMING == True
    assert config.VECTOR_STORE_TYPE == VectorStoreType.BM25S

def test_config_vector_store_types():
    """Test all vector store types including BM25S"""
    config = Config(VECTOR_STORE_TYPE=VectorStoreType.BM25S)
    assert config.VECTOR_STORE_TYPE == "bm25s"
    
    config = Config(VECTOR_STORE_TYPE=VectorStoreType.BM25)
    assert config.VECTOR_STORE_TYPE == "bm25"

def test_config_chunker_types():
    """Test all chunker types including image"""
    config = Config(CHUNKER_TYPE=ChunkerType.IMAGE)
    assert config.CHUNKER_TYPE == "image"

def test_config_environment_variables():
    """Test new environment variables"""
    os.environ["OCR_DEFAULT_MODE"] = "handwritten"
    os.environ["ENABLE_STREAMING"] = "true"
    
    config = Config()
    
    assert config.OCR_DEFAULT_MODE == "handwritten"
    assert config.ENABLE_STREAMING == True
    
    # Cleanup
    del os.environ["OCR_DEFAULT_MODE"]
    del os.environ["ENABLE_STREAMING"]


def test_comma_separated_env_lists(monkeypatch):
    """Regression: comma-separated env vars must not crash Config() (pydantic-settings JSON decoding)"""
    monkeypatch.setenv("CORS_ALLOW_METHODS", "GET,POST")
    monkeypatch.setenv("FALLBACK_PROVIDERS", "ollama, anthropic")
    
    config = Config()
    
    assert config.CORS_ALLOW_METHODS == ["GET", "POST"]
    assert config.FALLBACK_PROVIDERS == ["ollama", "anthropic"]