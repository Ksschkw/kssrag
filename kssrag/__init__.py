"""
KSS RAG - A flexible Retrieval-Augmented Generation framework by Ksschkw
"""
from .kssrag import KSSRAG
from .core.chunkers import TextChunker, JSONChunker, PDFChunker
from .core.vectorstores import BM25VectorStore, FAISSVectorStore, TFIDFVectorStore, HybridVectorStore, HybridOfflineVectorStore
from .core.retrievers import SimpleRetriever, HybridRetriever
from .core.agents import RAGAgent
from .models.openrouter import OpenRouterLLM
from .utils.document_loaders import load_document, load_json_documents
from .utils.helpers import logger, validate_config
from .config import Config, VectorStoreType, ChunkerType, RetrieverType
from .server import create_app, ServerConfig
from .cli import main

__version__ = "0.1.0"
__author__ = "Ksschkw"
__license__ = "MIT"

# Your footprint - include your GitHub username and a signature
__signature__ = "Built with HATE by Ksschkw (github.com/Ksschkw)"

# Export the main classes for easy access
__all__ = [
    'KSSRAG',
    'TextChunker',
    'JSONChunker',
    'PDFChunker',
    'BM25VectorStore',
    'FAISSVectorStore',
    'TFIDFVectorStore',
    'HybridVectorStore',
    'HybridOfflineVectorStore',
    'SimpleRetriever',
    'HybridRetriever',
    'RAGAgent',
    'OpenRouterLLM',
    'load_document',
    'load_json_documents',
    'Config',
    'VectorStoreType',
    'ChunkerType',
    'RetrieverType',
    'ServerConfig',
    'create_app',
    'main',
    'logger',
    'validate_config'
]

# Initialize configuration validation on import
validate_config()

import platform
from pathlib import Path
import os

# Windows-specific cache directory handling
if platform.system() == "Windows":
    # Use local app data directory instead of home directory
    cache_base = os.getenv('LOCALAPPDATA', os.path.expanduser('~'))
    config.CACHE_DIR = os.path.join(cache_base, '.kssrag', 'cache')
else:
    config.CACHE_DIR = os.path.join(os.path.expanduser('~'), '.cache', 'kssrag')

os.makedirs(config.CACHE_DIR, exist_ok=True)