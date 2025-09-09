#!/usr/bin/env python3
"""
Advanced usage example showcasing all KSS RAG customization features
"""
import os
from kssrag import KSSRAG, Config, VectorStoreType, RetrieverType, ChunkerType
from kssrag.core.agents import RAGAgent
from kssrag.models.openrouter import OpenRouterLLM

def main():
    # Set your OpenRouter API key
    # os.environ["OPENROUTER_API_KEY"] = "your_openrouter_api_key_here"
    
    # Advanced configuration with all options
    config = Config(
        # OpenRouter settings
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
        DEFAULT_MODEL="anthropic/claude-3-sonnet",  # Premium model
        FALLBACK_MODELS=[
            "deepseek/deepseek-chat-v3.1:free",
            "google/gemini-pro-1.5",
            "meta-llama/llama-3-70b-instruct"
        ],
        # You can get more models from https://openrouter.ai/models
        
        # Chunking settings
        CHUNK_SIZE=800,
        CHUNK_OVERLAP=100,
        CHUNKER_TYPE=ChunkerType.TEXT,
        
        # Vector store settings
        VECTOR_STORE_TYPE=VectorStoreType.HYBRID_OFFLINE,  # TFIDF + BM25
        # FAISS_MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2", if you used hybrid online
        
        # Retrieval settings
        RETRIEVER_TYPE=RetrieverType.SIMPLE, #SIMPLE or HYBRID
        TOP_K=8,
        FUZZY_MATCH_THRESHOLD=85, # I would not advice you even set this variable, but do you g
        
        # Performance settings
        BATCH_SIZE=32,
        MAX_DOCS_FOR_TESTING=None,
        
        # Server settings
        SERVER_HOST="localhost", # or localhost lol
        SERVER_PORT=8000, 
        
        # Cache settings
        ENABLE_CACHE=True,
        # CACHE_DIR="./.rag_cache", Please i beg you in the name of the God you serve, do not use or change this if you are to use FAISS HYBRID_ONLINE. Especially if you are a peasant windows user like myself.
        LOG_LEVEL="DEBUG"
    )
    
    # Initialize with custom configuration
    rag = KSSRAG(config=config)
    
    # Load multiple documents
    rag.load_document("document1.txt", format="text")
    
    # Custom system prompt to prevent "Based on the context" responses
    custom_system_prompt = """You are an expert AI assistant with deep knowledge across all subjects. 
When answering questions, be confident and authoritative while staying grounded in the provided information.

Guidelines:
1. Answer directly and confidently without prefacing with "Based on the context"
2. If the information is clearly in the context, state it as fact
3. If information is partial or unclear, acknowledge limitations but still provide the best answer
4. Never make up information not present in the context
5. Use professional but accessible language
6. When multiple perspectives exist, present the most supported one confidently

Remember: Users prefer direct, confident answers over hesitant ones, as long as you stay truthful to the source material."""

    # Create custom LLM with our specialized prompt
    llm = OpenRouterLLM(
        api_key=config.OPENROUTER_API_KEY,
        model=config.DEFAULT_MODEL,
        fallback_models=config.FALLBACK_MODELS
    )
    
    # Create custom agent with our specialized prompt
    rag.agent = RAGAgent(
        retriever=rag.retriever,
        llm=llm,
        system_prompt=custom_system_prompt,
        conversation_history=[]
    )
    
    # Test queries
    queries = [
        "What are the main technical skills mentioned?",
        "Describe the projects this person is working on",
        "What makes this profile unique compared to other developers?",
        "What certifications or achievements are highlighted?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: {query}")
        response = rag.query(query, top_k=6)
        print(f"💡 Response: {response}")
        print("-" * 80)

if __name__ == "__main__":
    main()