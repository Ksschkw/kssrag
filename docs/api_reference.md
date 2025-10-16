# API Reference

## Overview

This document provides comprehensive API reference for KSS RAG framework. All classes, methods, and configuration options are documented with examples and usage patterns.

## Core Classes

### KSSRAG

The main entry point for the RAG framework.

```python
class KSSRAG:
    """
    Main RAG framework class providing document processing and query capabilities.
    
    This class orchestrates the entire RAG pipeline including document loading,
    vector store management, retrieval, and response generation.
    
    Attributes:
        config (Config): Framework configuration instance
        vector_store: Active vector store implementation
        retriever: Document retriever instance
        agent: RAG agent for query processing
        documents (List[Dict]): Processed document chunks
    """
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize RAG system with optional configuration.
        
        Args:
            config: Configuration instance. If None, uses default configuration.
            
        Example:
            >>> from kssrag import KSSRAG
            >>> rag = KSSRAG()
            >>> # Or with custom config
            >>> from kssrag import Config
            >>> config = Config(TOP_K=10)
            >>> rag = KSSRAG(config=config)
        """
        
    def load_document(self, 
                     file_path: str, 
                     format: Optional[str] = None,
                     chunker: Optional[Any] = None, 
                     metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Load and process document for retrieval.
        
        Args:
            file_path: Path to the document file
            format: Document format. Auto-detected from extension if None.
                    Supported: 'text', 'json', 'pdf', 'image', 'docx', 'excel', 'pptx'
            chunker: Custom chunker instance. Uses default chunker if None.
            metadata: Additional document metadata to attach to chunks.
            
        Raises:
            FileNotFoundError: If file_path doesn't exist
            ValueError: If format is unsupported or auto-detection fails
            Exception: For document processing errors
            
        Example:
            >>> rag.load_document("document.pdf")
            >>> rag.load_document("data.json", format="json")
            >>> rag.load_document("scan.jpg", format="image", 
            ...                  metadata={"source": "scanned_doc"})
        """
        
    def query(self, 
              question: str, 
              top_k: Optional[int] = None) -> str:
        """
        Execute query against loaded documents.
        
        Args:
            question: Natural language query string
            top_k: Number of results to retrieve. Uses config.TOP_K if None.
            
        Returns:
            Generated response string from LLM
            
        Raises:
            ValueError: If no documents are loaded
            Exception: For retrieval or generation errors
            
        Example:
            >>> response = rag.query("What are the main features?")
            >>> response = rag.query("Explain the architecture", top_k=8)
        """
        
    def create_server(self, server_config=None) -> Tuple[FastAPI, Any]:
        """
        Create FastAPI server instance for the RAG system.
        
        Args:
            server_config: Custom server configuration
            
        Returns:
            Tuple of (FastAPI app, server_config)
            
        Example:
            >>> app, config = rag.create_server()
            >>> import uvicorn
            >>> uvicorn.run(app, host="0.0.0.0", port=8000)
        """
```

### Config

Configuration management with validation.

```python
class Config(BaseSettings):
    """
    Comprehensive configuration management with Pydantic validation.
    
    Supports environment variables, .env files, and programmatic configuration.
    All configuration options are validated with sensible defaults.
    
    Example:
        >>> from kssrag import Config
        >>> config = Config(
        ...     OPENROUTER_API_KEY="your-key",
        ...     VECTOR_STORE_TYPE=VectorStoreType.HYBRID_ONLINE,
        ...     TOP_K=10
        ... )
    """
    
    # API Configuration
    OPENROUTER_API_KEY: str = Field(
        default=...,
        description="OpenRouter API key for LLM access",
        min_length=1
    )
    
    DEFAULT_MODEL: str = Field(
        default="anthropic/claude-3-sonnet",
        description="Primary LLM model for responses"
    )
    
    FALLBACK_MODELS: List[str] = Field(
        default=["deepseek/deepseek-chat-v3.1:free"],
        description="Fallback models if primary fails"
    )
    
    # Document Processing
    CHUNK_SIZE: int = Field(
        default=800,
        ge=100,
        le=2000,
        description="Size of text chunks in characters"
    )
    
    CHUNK_OVERLAP: int = Field(
        default=100,
        ge=0,
        le=500,
        description="Overlap between chunks in characters"
    )
    
    CHUNKER_TYPE: ChunkerType = Field(
        default=ChunkerType.TEXT,
        description="Type of chunker to use"
    )
    
    # Vector Store Configuration
    VECTOR_STORE_TYPE: VectorStoreType = Field(
        default=VectorStoreType.HYBRID_OFFLINE,
        description="Type of vector store to use"
    )
    
    FAISS_MODEL_NAME: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="SentenceTransformer model for FAISS embeddings"
    )
    
    # Retrieval Configuration
    RETRIEVER_TYPE: RetrieverType = Field(
        default=RetrieverType.SIMPLE,
        description="Type of retriever to use"
    )
    
    TOP_K: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to retrieve"
    )
    
    FUZZY_MATCH_THRESHOLD: int = Field(
        default=80,
        ge=0,
        le=100,
        description="Threshold for fuzzy matching (0-100)"
    )
```

## Vector Stores

### BaseVectorStore

```python
class BaseVectorStore(ABC):
    """
    Abstract base class for vector store implementations.
    
    All vector stores must implement these methods for consistent interface.
    """
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """
        Add documents to the vector store.
        
        Args:
            documents: List of document dictionaries with 'content' and 'metadata'
        """
        
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on query.
        
        Args:
            query: Search query string
            top_k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        
    def persist(self) -> None:
        """Persist vector store to disk."""
        
    def load(self) -> None:
        """Load vector store from disk."""
```

### BM25VectorStore

```python
class BM25VectorStore(BaseVectorStore):
    """
    BM25 vector store for keyword-based retrieval.
    
    Uses rank_bm25 library for efficient keyword matching.
    Ideal for technical documentation and code search.
    
    Example:
        >>> from kssrag.core.vectorstores import BM25VectorStore
        >>> store = BM25VectorStore()
        >>> store.add_documents(documents)
        >>> results = store.retrieve("python programming", top_k=5)
    """
```

### FAISSVectorStore

```python
class FAISSVectorStore(BaseVectorStore):
    """
    FAISS vector store for semantic similarity search.
    
    Uses Facebook AI Similarity Search for efficient vector retrieval.
    Requires sentence-transformers for embeddings.
    
    Example:
        >>> store = FAISSVectorStore()
        >>> store.add_documents(documents)
        >>> results = store.retrieve("machine learning algorithms", top_k=5)
    """
```

### HybridVectorStore

```python
class HybridVectorStore(BaseVectorStore):
    """
    Hybrid vector store combining BM25 and FAISS.
    
    Provides balanced retrieval using both keyword and semantic search.
    Automatically handles score fusion and deduplication.
    
    Example:
        >>> store = HybridVectorStore()
        >>> store.add_documents(documents)
        >>> results = store.retrieve("complex query", top_k=8)
    """
```

## Retrievers

### BaseRetriever

```python
class BaseRetriever(ABC):
    """
    Abstract base class for document retrievers.
    """
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve documents based on query.
        
        Args:
            query: Search query
            top_k: Number of results
            
        Returns:
            List of relevant documents
        """
```

### SimpleRetriever

```python
class SimpleRetriever(BaseRetriever):
    """
    Simple retriever using only vector store.
    
    Direct passthrough to vector store retrieval.
    Fast and efficient for most use cases.
    
    Example:
        >>> from kssrag.core.retrievers import SimpleRetriever
        >>> retriever = SimpleRetriever(vector_store)
        >>> results = retriever.retrieve("query", top_k=5)
    """
```

### HybridRetriever

```python
class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever with fuzzy matching and entity extraction.
    
    Enhances retrieval with fuzzy matching for specific entities
    and query understanding capabilities.
    
    Example:
        >>> retriever = HybridRetriever(vector_store, entity_names)
        >>> results = retriever.retrieve("query with entities", top_k=5)
    """
```

## Agents

### RAGAgent

```python
class RAGAgent:
    """
    RAG agent for orchestrating retrieval and generation.
    
    Handles conversation management, context building, and LLM interaction.
    Supports both standard and streaming responses.
    
    Example:
        >>> from kssrag.core.agents import RAGAgent
        >>> agent = RAGAgent(retriever, llm, system_prompt="You are an expert.")
        >>> response = agent.query("What is this about?")
    """
    
    def __init__(self, 
                 retriever: BaseRetriever,
                 llm: Any,
                 system_prompt: Optional[str] = None,
                 conversation_history: Optional[List[Dict[str, str]]] = None):
        """
        Initialize RAG agent.
        
        Args:
            retriever: Document retriever instance
            llm: LLM provider instance
            system_prompt: Custom system prompt for LLM
            conversation_history: Previous conversation history
        """
        
    def query(self, 
              question: str, 
              top_k: int = 5,
              include_context: bool = True) -> str:
        """
        Process query and return response.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            include_context: Whether to include retrieved context
            
        Returns:
            LLM generated response
        """
        
    def query_stream(self, 
                     question: str, 
                     top_k: int = 5) -> Generator[str, None, None]:
        """
        Process query with streaming response.
        
        Args:
            question: User question
            top_k: Number of documents to retrieve
            
        Yields:
            Response chunks as they are generated
        """
        
    def clear_conversation(self) -> None:
        """Clear conversation history except system message."""
        
    def add_message(self, role: str, content: str) -> None:
        """
        Add message to conversation history.
        
        Args:
            role: Message role ('user', 'assistant', 'system')
            content: Message content
        """
```

## Document Loaders and Chunkers

### BaseChunker

```python
class BaseChunker(ABC):
    """
    Abstract base class for document chunkers.
    """
    
    @abstractmethod
    def chunk(self, 
              content: Any, 
              metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk content into documents.
        
        Args:
            content: Input content to chunk
            metadata: Additional metadata for chunks
            
        Returns:
            List of chunk dictionaries
        """
```

### TextChunker

```python
class TextChunker(BaseChunker):
    """
    Chunker for plain text documents.
    
    Implements sliding window chunking with configurable
    chunk size and overlap.
    
    Example:
        >>> chunker = TextChunker(chunk_size=500, overlap=50)
        >>> chunks = chunker.chunk(long_text, metadata={"source": "doc.txt"})
    """
```

### PDFChunker

```python
class PDFChunker(TextChunker):
    """
    Chunker for PDF documents.
    
    Extends TextChunker with PDF text extraction capabilities.
    
    Example:
        >>> chunker = PDFChunker()
        >>> chunks = chunker.chunk_pdf("document.pdf", metadata={"type": "pdf"})
    """
```

### ImageChunker

```python
class ImageChunker(BaseChunker):
    """
    Chunker for image documents using OCR.
    
    Supports both handwritten (PaddleOCR) and typed (Tesseract) text.
    
    Example:
        >>> chunker = ImageChunker(ocr_mode="handwritten")
        >>> chunks = chunker.chunk("scan.jpg", metadata={"scanned": True})
    """
```

## LLM Providers

### OpenRouterLLM

```python
class OpenRouterLLM:
    """
    OpenRouter LLM provider with streaming support.
    
    Provides access to 100+ language models through OpenRouter API.
    Includes smart fallbacks and comprehensive error handling.
    
    Example:
        >>> from kssrag.models.openrouter import OpenRouterLLM
        >>> llm = OpenRouterLLM(api_key="key", model="anthropic/claude-3-sonnet")
        >>> response = llm.predict(messages)
    """
    
    def predict(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate response with fallback models.
        
        Args:
            messages: Conversation messages
            
        Returns:
            Generated response text
        """
        
    def predict_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        Stream response from OpenRouter.
        
        Args:
            messages: Conversation messages
            
        Yields:
            Response chunks as they are generated
        """
```

## Utility Functions

### Document Loaders

```python
def load_document(file_path: str) -> str:
    """
    Load document from file.
    
    Args:
        file_path: Path to document file
        
    Returns:
        Document content as string
    """

def load_json_documents(file_path: str, metadata_field: str = "name") -> List[Dict[str, Any]]:
    """
    Load documents from JSON file.
    
    Args:
        file_path: Path to JSON file
        metadata_field: Field to use for document metadata
        
    Returns:
        List of document dictionaries
    """
```

### OCR Loader

```python
class OCRLoader:
    """
    Production OCR handler with PaddleOCR and Tesseract.
    
    Provides unified interface for OCR processing with
    mode-specific optimizations.
    
    Example:
        >>> from kssrag.utils.ocr_loader import OCRLoader
        >>> loader = OCRLoader()
        >>> text = loader.extract_text("image.jpg", mode="handwritten")
    """
    
    def extract_text(self, image_path: str, mode: str = "typed") -> str:
        """
        Extract text from image using specified OCR engine.
        
        Args:
            image_path: Path to image file
            mode: OCR mode ('typed' or 'handwritten')
            
        Returns:
            Extracted text
        """
```

## Server API Endpoints

### Query Endpoint

```python
@app.post("/query")
async def query_endpoint(request: QueryRequest):
    """
    Handle user queries.
    
    Request:
        {
            "query": "user question",
            "session_id": "optional session id"
        }
    
    Response:
        {
            "query": "original query",
            "response": "generated response", 
            "session_id": "session identifier"
        }
    """
```

### Stream Endpoint

```python
@app.post("/stream") 
async def stream_query(request: QueryRequest):
    """
    Streaming query endpoint with Server-Sent Events.
    
    Returns streaming response with chunked data.
    """
```

### Health Check

```python
@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Response:
        {
            "status": "healthy",
            "message": "KSS RAG API is running",
            "version": "0.2.0"
        }
    """
```

## Error Handling

### Common Exceptions

```python
class KSSRAGError(Exception):
    """Base exception for KSS RAG framework."""
    pass

class ConfigurationError(KSSRAGError):
    """Configuration-related errors."""
    pass

class DocumentProcessingError(KSSRAGError):
    """Document loading and processing errors."""
    pass

class VectorStoreError(KSSRAGError):
    """Vector store operation errors."""
    pass

class LLMError(KSSRAGError):
    """LLM API and generation errors."""
    pass
```

### Error Recovery

```python
try:
    rag.load_document("document.pdf")
    response = rag.query("question")
except DocumentProcessingError as e:
    print(f"Document error: {e}")
    # Implement fallback logic
except LLMError as e:
    print(f"LLM error: {e}") 
    # Implement retry or alternative response
```

This API reference provides comprehensive documentation for all KSS RAG components. For specific implementation details or advanced usage, refer to the source code or examples directory.

---

