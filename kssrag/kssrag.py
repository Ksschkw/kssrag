"""
Main KSSRAG class that ties everything together for easy usage.
"""
from typing import Optional, List, Dict, Any
import os
from .core.chunkers import TextChunker, JSONChunker, PDFChunker
from .core.vectorstores import BM25VectorStore, BM25SVectorStore, FAISSVectorStore, TFIDFVectorStore, HybridVectorStore, HybridOfflineVectorStore
from .core.retrievers import SimpleRetriever, HybridRetriever
from .core.agents import RAGAgent
from .models.openrouter import OpenRouterLLM
from .models.factory import create_llm
from .utils.document_loaders import load_document, load_json_documents
from .config import Config, VectorStoreType, ChunkerType, RetrieverType
from .utils.helpers import logger, validate_config, import_custom_component

class KSSRAG:
    """Main class for KSS RAG functionality"""
    
    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()
        self.vector_store = None
        self.retriever = None
        self.agent = None
        self.documents = []
        
        # Validate configuration
        validate_config()
    
    def load_document(self, file_path: str, format: Optional[str] = None, 
                     chunker: Optional[Any] = None, metadata: Optional[Dict[str, Any]] = None):
        """Load and process a document"""
        if format is None:
            # Auto-detect format
            if file_path.endswith('.txt'):
                format = 'text'
            elif file_path.endswith('.json'):
                format = 'json'
            elif file_path.endswith('.pdf'):
                format = 'pdf'
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
        
        # Use custom chunker if provided
        if chunker is None:
            if format == 'text':
                chunker = TextChunker(chunk_size=self.config.CHUNK_SIZE, overlap=self.config.CHUNK_OVERLAP)
            elif format == 'json':
                chunker = JSONChunker()
            elif format == 'pdf':
                chunker = PDFChunker(chunk_size=self.config.CHUNK_SIZE, overlap=self.config.CHUNK_OVERLAP)
        
        if metadata is None:
            metadata = {"source": file_path}

        # Load and chunk document (deferred so caching can skip it on a hit)
        def _build_documents():
            if format == 'text':
                content = load_document(file_path)
                return chunker.chunk(content, metadata)
            elif format == 'json':
                data = load_json_documents(file_path)
                return chunker.chunk(data)
            elif format == 'pdf':
                return chunker.chunk_pdf(file_path, metadata)
            raise ValueError(f"Unsupported file format: {file_path}")

        # Create vector store
        if self.config.CUSTOM_VECTOR_STORE:
            vector_store_class = import_custom_component(self.config.CUSTOM_VECTOR_STORE)
            self.vector_store = vector_store_class()
            self.documents = _build_documents()
            self.vector_store.add_documents(self.documents)
        elif self.config.ENABLE_CACHE:
            from .utils.cache import load_or_build_index
            self.vector_store, self.documents = load_or_build_index(
                source_path=file_path,
                store_type=self.config.VECTOR_STORE_TYPE,
                build_documents=_build_documents,
                cache_root=self.config.CACHE_DIR,
                chunk_size=self.config.CHUNK_SIZE,
                chunk_overlap=self.config.CHUNK_OVERLAP,
                max_docs=self.config.MAX_DOCS_FOR_TESTING,
            )
        else:
            if self.config.VECTOR_STORE_TYPE == VectorStoreType.BM25:
                self.vector_store = BM25VectorStore()
            elif self.config.VECTOR_STORE_TYPE == VectorStoreType.BM25S:
                self.vector_store = BM25SVectorStore()
            elif self.config.VECTOR_STORE_TYPE == VectorStoreType.FAISS:
                self.vector_store = FAISSVectorStore()
            elif self.config.VECTOR_STORE_TYPE == VectorStoreType.TFIDF:
                self.vector_store = TFIDFVectorStore()
            elif self.config.VECTOR_STORE_TYPE == VectorStoreType.HYBRID_ONLINE:
                self.vector_store = HybridVectorStore()
            elif self.config.VECTOR_STORE_TYPE == VectorStoreType.HYBRID_OFFLINE:
                self.vector_store = HybridOfflineVectorStore()
            else:
                raise ValueError(
                    f"Unsupported vector store type: {self.config.VECTOR_STORE_TYPE}"
                )
            self.documents = _build_documents()
            self.vector_store.add_documents(self.documents)

        # Create retriever
        if self.config.CUSTOM_RETRIEVER:
            retriever_class = import_custom_component(self.config.CUSTOM_RETRIEVER)
            self.retriever = retriever_class(self.vector_store)
        else:
            if self.config.RETRIEVER_TYPE == RetrieverType.SIMPLE:
                self.retriever = SimpleRetriever(self.vector_store)
            elif self.config.RETRIEVER_TYPE == RetrieverType.HYBRID:
                # For hybrid retriever, extract entity names from documents if available
                entity_names = []
                if format == 'json':
                    entity_names = [doc['metadata'].get('name', '') for doc in self.documents if doc['metadata'].get('name')]
                self.retriever = HybridRetriever(self.vector_store, entity_names)
        
        # Create LLM. Building the index shouldn't require an LLM key (retrieval
        # works without one), so a missing-key error is deferred to query time.
        if self.config.CUSTOM_LLM:
            llm_class = import_custom_component(self.config.CUSTOM_LLM)
            llm = llm_class()
        else:
            try:
                llm = create_llm(
                    provider=self.config.PROVIDER,
                    model=self.config.DEFAULT_MODEL,
                    fallback_models=self.config.FALLBACK_MODELS,
                    cfg=self.config,
                )
            except ValueError as e:
                logger.warning(f"LLM not configured yet: {e}")
                llm = None
        
        # Create agent
        self.agent = RAGAgent(self.retriever, llm)
    
    def query(self, question: str, top_k: Optional[int] = None) -> str:
        """Query the RAG system"""
        if not self.agent:
            raise ValueError("No documents loaded. Call load_document first.")
        
        return self.agent.query(question, top_k=top_k or self.config.TOP_K)
    
    def create_server(self, server_config=None):
        """Create a FastAPI server for the RAG system"""
        from .server import create_app
        return create_app(self.agent, server_config)