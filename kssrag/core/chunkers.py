import json
import re
from typing import List, Dict, Any, Optional
import pypdf
from ..utils.helpers import logger
import os
try:
    from ..utils.ocr_loader import OCRLoader
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    OCRLoader = None

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
    
class ImageChunker(BaseChunker):
    """Chunker for image documents using OCR with PaddleOCR and Tesseract"""
    
    def __init__(self, chunk_size: int = 500, overlap: int = 50, ocr_mode: str = "typed"):
        super().__init__(chunk_size, overlap)
        self.ocr_mode = ocr_mode  # typed or handwritten
        self.ocr_loader = None
        
        # Initialize OCR loader
        try:
            from ..utils.ocr_loader import OCRLoader
            self.ocr_loader = OCRLoader()
            logger.info(f"OCR loader initialized with mode: {ocr_mode}")
        except ImportError as e:
            logger.error(f"OCR dependencies not available: {str(e)}")
            raise ImportError(
                "OCR functionality requires extra dependencies. "
                "Install with: pip install kssrag[ocr]"
            ) from e
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using specified OCR engine"""
        if not self.ocr_loader:
            raise RuntimeError("OCR loader not initialized")
        
        if self.ocr_mode not in ["typed", "handwritten"]:
            raise ValueError(f"Invalid OCR mode: {self.ocr_mode}. Must be 'typed' or 'handwritten'")
        
        logger.info(f"Extracting text from {image_path} using {self.ocr_mode} OCR")
        
        try:
            text = self.ocr_loader.extract_text(image_path, self.ocr_mode)
            
            if not text.strip():
                logger.warning(f"No text extracted from image: {image_path}")
                return ""
            
            logger.info(f"Successfully extracted {len(text)} characters from {image_path}")
            return text
            
        except Exception as e:
            logger.error(f"OCR extraction failed for {image_path}: {str(e)}")
            raise RuntimeError(f"Failed to extract text from image {image_path}: {str(e)}")
    
    def chunk(self, image_path: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Extract text from image and chunk it"""
        if metadata is None:
            metadata = {}
        
        # Validate image file
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Extract text from image
        text = self.extract_text_from_image(image_path)
        
        if not text.strip():
            return []
        
        # Use text chunking on extracted text
        text_chunker = TextChunker(chunk_size=self.chunk_size, overlap=self.overlap)
        chunks = text_chunker.chunk(text, metadata)
        
        # Add OCR-specific metadata
        for chunk in chunks:
            chunk["metadata"]["ocr_extracted"] = True
            chunk["metadata"]["image_source"] = image_path
            chunk["metadata"]["ocr_mode"] = self.ocr_mode
        
        logger.info(f"Created {len(chunks)} chunks from image {image_path}")
        return chunks