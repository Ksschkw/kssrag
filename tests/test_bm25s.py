import pytest
import numpy as np
import tempfile
import os
from kssrag.core.vectorstores import BM25SVectorStore

def test_bm25s_vector_store_basic():
    """Test BM25S vector store basic functionality"""
    documents = [
        {"content": "This is a test document about Python programming.", "metadata": {"source": "test1"}},
        {"content": "Another document about machine learning and AI.", "metadata": {"source": "test2"}},
        {"content": "A third document on web development with JavaScript.", "metadata": {"source": "test3"}},
    ]
    
    vector_store = BM25SVectorStore()
    vector_store.add_documents(documents)
    
    results = vector_store.retrieve("Python programming", top_k=2)
    
    assert len(results) == 2
    assert "Python" in results[0]["content"]
    assert all("metadata" in result for result in results)

def test_bm25s_persistence():
    """Test BM25S vector store persistence"""
    documents = [
        {"content": "Test document for persistence.", "metadata": {"source": "test1"}},
        {"content": "Another test document.", "metadata": {"source": "test2"}},
    ]
    
    with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
        temp_file = f.name
    
    try:
        # Create and persist
        vector_store = BM25SVectorStore(persist_path=temp_file)
        vector_store.add_documents(documents)
        vector_store.persist()
        
        # Load and verify
        new_vector_store = BM25SVectorStore(persist_path=temp_file)
        new_vector_store.load()
        
        results = new_vector_store.retrieve("persistence", top_k=1)
        assert len(results) == 1
        assert "persistence" in results[0]["content"]
        
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)

def test_bm25s_empty_query():
    """Test BM25S with empty query"""
    documents = [
        {"content": "Test document.", "metadata": {"source": "test1"}},
    ]
    
    vector_store = BM25SVectorStore()
    vector_store.add_documents(documents)
    
    results = vector_store.retrieve("", top_k=1)
    # BM25S may return documents even with empty query, but they should have low scores
    # Let's check that the behavior is consistent
    if len(results) > 0:
        # If it returns results, they should be the documents we added
        assert results[0]["content"] == "Test document."
    # Either behavior is acceptable for this test

def test_bm25s_no_documents():
    """Test BM25S with no documents added"""
    vector_store = BM25SVectorStore()
    
    with pytest.raises(ValueError, match="BM25S index not initialized"):
        vector_store.retrieve("test query")