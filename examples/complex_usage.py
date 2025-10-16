#!/usr/bin/env python3
"""
Advanced usage example showcasing all KSS RAG customization features
"""
import os
from kssrag import KSSRAG, Config, VectorStoreType, RetrieverType
from kssrag.core.agents import RAGAgent
from kssrag.models.openrouter import OpenRouterLLM

def main():
    # Advanced configuration with all options from your actual Config class
    config = Config(
        # OpenRouter settings
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
        DEFAULT_MODEL="anthropic/claude-3-sonnet",
        FALLBACK_MODELS=["deepseek/deepseek-chat-v3.1:free", "google/gemini-pro-1.5"],
        
        # Chunking settings
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        
        # Vector store settings
        VECTOR_STORE_TYPE=VectorStoreType.HYBRID_ONLINE,
        FAISS_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2",
        
        # Retrieval settings
        RETRIEVER_TYPE=RetrieverType.HYBRID,
        TOP_K=8,
        FUZZY_MATCH_THRESHOLD=85,
        
        # Performance settings
        BATCH_SIZE=32,
        
        # OCR settings
        OCR_DEFAULT_MODE="typed",
        ENABLE_STREAMING=True,
        
        # Server settings
        SERVER_HOST="localhost",
        SERVER_PORT=8000
    )
    
    # Initialize with custom configuration
    rag = KSSRAG(config=config)
    
    # Load multiple document types
    rag.load_document("document1.txt", format="text")
    rag.load_document("data.json", format="json")
    rag.load_document("scanned_doc.jpg", format="image")
    rag.load_document("presentation.pptx", format="pptx")
    
    # Custom system prompt
    custom_system_prompt = """You are an expert AI assistant. Answer questions confidently 
    and directly without prefacing with "Based on the context". Be authoritative 
    while staying truthful to the source material."""
    
    # Create custom LLM with streaming
    llm = OpenRouterLLM(
        api_key=config.OPENROUTER_API_KEY,
        model=config.DEFAULT_MODEL,
        stream=True
    )
    
    # Create custom agent
    rag.agent = RAGAgent(
        retriever=rag.retriever,
        llm=llm,
        system_prompt=custom_system_prompt
    )
    
    # Test queries with streaming
    queries = [
        "What are the main topics covered?",
        "Summarize the key findings",
        "What technical specifications are mentioned?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("Response: ", end="", flush=True)
        
        # Use streaming if enabled
        if config.ENABLE_STREAMING:
            for chunk in rag.agent.query_stream(query, top_k=6):
                print(chunk, end="", flush=True)
            print()  # New line after streaming
        else:
            response = rag.query(query, top_k=6)
            print(response)

if __name__ == "__main__":
    main()