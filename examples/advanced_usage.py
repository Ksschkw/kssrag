#!/usr/bin/env python3
"""
Advanced usage example for KSS RAG with custom configuration
"""
import os
from kssrag import KSSRAG, Config, VectorStoreType, RetrieverType

def main():
    # Custom configuration
    config = Config(
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
        DEFAULT_MODEL="anthropic/claude-3-sonnet",
        VECTOR_STORE_TYPE=VectorStoreType.HYBRID_OFFLINE,
        RETRIEVER_TYPE=RetrieverType.SIMPLE,
        TOP_K=10,
        CHUNK_SIZE=1000,
        CHUNK_OVERLAP=100
    )
    
    # Initialize with custom configuration
    rag = KSSRAG(config=config)
    
    # Load multiple documents
    rag.load_document("document1.txt")
    # rag.load_document("document2.pdf")
    # rag.load_document("data.json", format="json")
    
    # Query the system
    response = rag.query("What are the key points across all documents?")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()