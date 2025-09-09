"""
Text preprocessing utilities for document preparation
"""
import re
from typing import List
from .helpers import logger

def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and special characters"""
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters except basic punctuation
    text = re.sub(r'[^\w\s.,!?;:]', '', text)
    return text.strip()

def split_sentences(text: str) -> List[str]:
    """Split text into sentences"""
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

def preprocess_document(text: str, clean: bool = True, split: bool = False) -> str | List[str]:
    """Preprocess a document with various options"""
    if clean:
        text = clean_text(text)
    
    if split:
        return split_sentences(text)
    
    return text