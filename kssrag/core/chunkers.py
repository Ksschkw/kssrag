import json
import re
from typing import List, Dict, Any, Optional
import pypdf
from ..utils.helpers import logger

class BaseChunker:
    """Base class for document chunkers"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk(self, content: Any, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement this method")

class TextChunker(BaseChunker):
    """Chunker for plain text documents"""
    
    def chunk(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Split text into chunks with overlap"""
        if metadata is None:
            metadata = {}
        
        chunks = []
        start = 0
        content_length = len(content)
        
        while start < content_length:
            end = start + self.chunk_size
            if end > content_length:
                end = content_length
            
            chunk = content[start:end]
            chunk_metadata = metadata.copy()
            chunk_metadata["chunk_id"] = len(chunks)
            
            chunks.append({
                "content": chunk,
                "metadata": chunk_metadata
            })
            
            start += self.chunk_size - self.overlap
        
        logger.info(f"Created {len(chunks)} chunks from text")
        return chunks

class JSONChunker(BaseChunker):
    """Chunker for JSON documents (like your drug data)"""
    
    def chunk(self, data: List[Dict[str, Any]], metadata_field: str = "name") -> List[Dict[str, Any]]:
        """Create chunks from JSON data"""
        chunks = []
        
        for item in data:
            if metadata_field not in item:
                continue
                
            # Create a comprehensive text representation
            item_text = f"Item Name: {item.get(metadata_field, 'N/A')}\n"
            
            for key, value in item.items():
                if key != metadata_field and value:
                    if isinstance(value, str):
                        item_text += f"{key.replace('_', ' ').title()}: {value}\n"
                    elif isinstance(value, list):
                        item_text += f"{key.replace('_', ' ').title()}: {', '.join(value)}\n"
            
            chunks.append({
                "content": item_text,
                "metadata": {
                    "name": item[metadata_field],
                    "source": item.get('url', 'N/A')
                }
            })
        
        logger.info(f"Created {len(chunks)} chunks from JSON data")
        return chunks

class PDFChunker(TextChunker):
    """Chunker for PDF documents"""
    
    def extract_text(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = pypdf.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {str(e)}")
            raise
        
        return text
    
    def chunk_pdf(self, pdf_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract text from PDF and chunk it"""
        text = self.extract_text(pdf_path)
        return self.chunk(text, metadata)