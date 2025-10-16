#!/usr/bin/env python3
"""
Example showing how to use custom components with KSS RAG
Note: This is for advanced users who want to extend the framework
"""
import os
from kssrag import KSSRAG, Config

def main():
    # Configuration with custom components (if you create them)
    config = Config(
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
        # Example custom component paths (you would need to create these)
        # CUSTOM_CHUNKER="my_custom_chunker.MyCustomChunker",
        # CUSTOM_VECTOR_STORE="my_custom_vectorstore.MyCustomVectorStore",
        # CUSTOM_LLM="my_custom_llm.MyCustomLLM",
        
        # Use built-in components with custom settings
        VECTOR_STORE_TYPE="bm25s",  # Using your BM25S implementation
        CHUNK_SIZE=1000,
        TOP_K=8,
        OCR_DEFAULT_MODE="handwritten"  # Using PaddleOCR for handwritten text
    )
    
    # Initialize with custom configuration
    rag = KSSRAG(config=config)
    
    # Load document with custom settings
    rag.load_document("document.txt")
    
    # Query the system
    response = rag.query("What is this document about?")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()

"""
If you came to read this that means that you hate me
USE THIS AT YOUR OWN PERIL
IF YOU PLAN ON USING THIS, YOU MIGHT AS WELL CREATE A CUSTOM RAG WITHOUT ME
"""