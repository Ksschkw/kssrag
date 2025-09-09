import pytest
import numpy as np
from kssrag.core.vectorstores import BM25VectorStore, TFIDFVectorStore

def test_bm25_vector_store():
    """Test BM25 vector store functionality"""
    documents = [
        {"content": "This is a test document about Python programming.", "metadata": {"source": "test1"}},
        {"content": "Another document about machine learning and AI.", "metadata": {"source": "test2"}},
        {"content": "A third document on web development with JavaScript.", "metadata": {"source": "test3"}},
    ]
    
    vector_store = BM25VectorStore()
    vector_store.add_documents(documents)
    
    results = vector_store.retrieve("Python programming", top_k=2)
    
    assert len(results) == 2
    assert "Python" in results[0]["content"]

def test_tfidf_vector_store():
    """Test TFIDF vector store functionality"""
    documents = [
        {"content": "This is a test document about Python programming.", "metadata": {"source": "test1"}},
        {"content": "Another document about machine learning and AI.", "metadata": {"source": "test2"}},
        {"content": "A third document on web development with JavaScript.", "metadata": {"source": "test3"}},
    ]
    
    vector_store = TFIDFVectorStore()
    vector_store.add_documents(documents)
    
    results = vector_store.retrieve("machine learning", top_k=2)
    
    assert len(results) == 2
    assert "machine learning" in results[0]["content"].lower()