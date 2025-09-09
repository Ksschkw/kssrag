#!/usr/bin/env python3
"""
Basic usage example for KSS RAG
"""
import os
from kssrag import KSSRAG

def main():
    # Set your OpenRouter API key (or set it in your .env file)
    os.environ["OPENROUTER_API_KEY"] = "your_api_key_here"
    
    # Initialize with default settings
    rag = KSSRAG()
    
    # Load a text document
    rag.load_document("sample.txt")
    
    # Query the system
    response = rag.query("What is this document about?")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()