import argparse
import sys
import os  # Add this import if not already present
from .utils.document_loaders import load_document, load_json_documents
from .core.chunkers import ImageChunker, OfficeChunker, TextChunker, JSONChunker, PDFChunker
from .core.vectorstores import BM25SVectorStore, BM25VectorStore, FAISSVectorStore, TFIDFVectorStore, HybridVectorStore, HybridOfflineVectorStore
from .core.retrievers import SimpleRetriever, HybridRetriever
from .core.agents import RAGAgent
from .models.openrouter import OpenRouterLLM
from .config import config
from .utils.helpers import logger, validate_config

def main():
    """Command-line interface for KSS RAG"""
    parser = argparse.ArgumentParser(description="KSS RAG - Retrieval-Augmented Generation Framework")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Query the RAG system")
    query_parser.add_argument("--file", type=str, required=True, help="Path to document file")
    query_parser.add_argument("--query", type=str, required=True, help="Query to ask")
    query_parser.add_argument("--format", type=str, default="text", 
                         choices=["text", "json", "pdf", "image", "docx", "excel", "pptx"],
                         help="Document format")
    query_parser.add_argument("--vector-store", type=str, default=config.VECTOR_STORE_TYPE,
                         choices=["bm25", "bm25s", "faiss", "tfidf", "hybrid_online", "hybrid_offline"],
                         help="Vector store type")
    query_parser.add_argument("--stream", action="store_true",
                         help="Enable streaming response")
    query_parser.add_argument("--top-k", type=int, default=config.TOP_K, help="Number of results to retrieve")
    query_parser.add_argument("--system-prompt", type=str, help="Path to a file containing the system prompt or the prompt text itself")
    query_parser.add_argument("--ocr-mode", type=str, choices=["typed", "handwritten"], 
                         default=config.OCR_DEFAULT_MODE,
                         help="OCR mode for image processing")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Start the RAG API server")
    server_parser.add_argument("--file", type=str, required=True, help="Path to document file")
    server_parser.add_argument("--format", type=str, default="text", 
                          choices=["text", "json", "pdf", "image", "docx", "excel", "pptx"],
                          help="Document format")
    # I Updated the server parser vector store choices
    server_parser.add_argument("--vector-store", type=str, default=config.VECTOR_STORE_TYPE,
                            choices=["bm25", "bm25s", "faiss", "tfidf", "hybrid_online", "hybrid_offline"],  # Add bm25s
                            help="Vector store type")
    server_parser.add_argument("--port", type=int, default=config.SERVER_PORT, help="Port to run server on")
    server_parser.add_argument("--host", type=str, default=config.SERVER_HOST, help="Host to run server on")
    server_parser.add_argument("--system-prompt", type=str, help="Path to a file containing the system prompt or the prompt text itself")
    
    args = parser.parse_args()
    vector_store_type = args.vector_store if hasattr(args, 'vector_store') else config.VECTOR_STORE_TYPE
    
    # Validate config
    validate_config()
    
    def load_system_prompt(prompt_arg):
        """Load system prompt from file or use as text"""
        if not prompt_arg:
            return None
        if os.path.exists(prompt_arg):
            with open(prompt_arg, 'r', encoding='utf-8') as f:
                return f.read()
        return prompt_arg
        
    
    if args.command == "query":
        # Load and process document
        if args.format == "text":
            content = load_document(args.file)
            chunker = TextChunker(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
            documents = chunker.chunk(content, {"source": args.file})
        elif args.format == "json":
            data = load_json_documents(args.file)
            chunker = JSONChunker()
            documents = chunker.chunk(data)
        elif args.format == "pdf":
            chunker = PDFChunker(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
            documents = chunker.chunk_pdf(args.file, {"source": args.file})
        elif args.format == "image":
            chunker = ImageChunker(
                chunk_size=config.CHUNK_SIZE, 
                overlap=config.CHUNK_OVERLAP,
                ocr_mode=getattr(args, 'ocr_mode', config.OCR_DEFAULT_MODE)
            )
            documents = chunker.chunk(args.file, {"source": args.file})
        elif args.format in ["docx", "excel", "pptx"]:
            # Use OfficeChunker for office documents
            chunker = OfficeChunker(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
            documents = chunker.chunk_office(args.file, {"source": args.file})
        else:
            logger.error(f"Unsupported format: {args.format}")
            return 1
        
        # Create vector store
        if args.vector_store == "bm25":
            vector_store = BM25VectorStore()
        elif args.vector_store == "faiss":
            vector_store = FAISSVectorStore()
        elif args.vector_store == "tfidf":
            vector_store = TFIDFVectorStore()
        elif args.vector_store == "hybrid_online":
            vector_store = HybridVectorStore()
        elif args.vector_store == "hybrid_offline":
            vector_store = HybridOfflineVectorStore()
        elif args.vector_store == "bm25s":
            vector_store = BM25SVectorStore()
        else:
            logger.error(f"Unsupported vector store: {args.vector_store}")
            return 1
        
        vector_store.add_documents(documents)
        
        # Create retriever and agent
        retriever = SimpleRetriever(vector_store)
        llm = OpenRouterLLM()
        system_prompt = load_system_prompt(args.system_prompt)
        agent = RAGAgent(retriever, llm, system_prompt=system_prompt)
        
        # Query and print response
        # response = agent.query(args.query, top_k=args.top_k)
        # print(f"Query: {args.query}")
        # print(f"Response: {response}")

        # In the query section, after creating the agent:
        if args.stream:
            print(f"Query: {args.query}")
            print("Response: ", end="", flush=True)
            
            try:
                # Collect all chunks and print them as they come
                full_response = ""
                for chunk in agent.query_stream(args.query, top_k=args.top_k):
                    print(chunk, end="", flush=True)
                    full_response += chunk
                print()  # New line at the end
                
                # The response is already added to conversation in query_stream
            except Exception as e:
                print(f"\nError during streaming: {str(e)}")
        else:
            response = agent.query(args.query, top_k=args.top_k)
            print(f"Query: {args.query}")
            print(f"Response: {response}")
        
    elif args.command == "server":
        # Load and process document
        if args.format == "text":
            content = load_document(args.file)
            chunker = TextChunker(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
            documents = chunker.chunk(content, {"source": args.file})
        elif args.format == "json":
            data = load_json_documents(args.file)
            chunker = JSONChunker()
            documents = chunker.chunk(data)
        elif args.format == "pdf":
            chunker = PDFChunker(chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
            documents = chunker.chunk_pdf(args.file, {"source": args.file})
        else:
            logger.error(f"Unsupported format: {args.format}")
            return 1
        
        # Create vector store
        if args.vector_store == "bm25":
            vector_store = BM25VectorStore()
        elif args.vector_store == "faiss":
            vector_store = FAISSVectorStore()
        elif args.vector_store == "tfidf":
            vector_store = TFIDFVectorStore()
        elif args.vector_store == "hybrid_online":
            vector_store = HybridVectorStore()
        elif args.vector_store == "hybrid_offline":
            vector_store = HybridOfflineVectorStore()
        elif args.vector_store == "bm25s":
            vector_store = BM25SVectorStore()
        else:
            logger.error(f"Unsupported vector store: {args.vector_store}")
            return 1
        
        vector_store.add_documents(documents)
        
        # Create retriever and agent
        retriever = SimpleRetriever(vector_store)
        llm = OpenRouterLLM()
        system_prompt = load_system_prompt(args.system_prompt)
        agent = RAGAgent(retriever, llm, system_prompt=system_prompt)
        
        # Create and run server
        from .server import create_app
        import uvicorn
        
        app, server_config = create_app(agent)
        logger.info(f"Starting server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
        
    else:
        parser.print_help()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())