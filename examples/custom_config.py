#!/usr/bin/env python3
"""
Example showing how to use custom components with KSS RAG
If you came to read this that means that you hate me
USE THIS AT YOUR OWN PERIL
IF YOU PLAN ON USING THIS, YOU MIGHT AS WELL CREATE A CUSTOM RAG WITHOUT ME
"""
import os
from kssrag import KSSRAG, Config

def main():
    # Configuration with custom components
    config = Config(
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
        CUSTOM_CHUNKER="my_custom_chunker.MyCustomChunker",
        CUSTOM_VECTOR_STORE="my_custom_vectorstore.MyCustomVectorStore",
        CUSTOM_LLM="my_custom_llm.MyCustomLLM"
    )
    
    # Initialize with custom configuration
    rag = KSSRAG(config=config)
    
    # Load document with custom chunker
    rag.load_document("document.txt")
    
    # Query the system
    response = rag.query("What is this document about?")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()