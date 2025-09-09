import json
from typing import List, Dict, Any, Optional
from ..utils.helpers import logger

def load_txt_file(file_path: str) -> str:
    """Load text from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Failed to load text file: {str(e)}")
        raise

def load_json_file(file_path: str) -> Any:
    """Load JSON from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON file: {str(e)}")
        raise

def load_document(file_path: str) -> str:
    """Load document from file (supports .txt)"""
    if file_path.endswith('.txt'):
        return load_txt_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def load_json_documents(file_path: str, metadata_field: str = "name") -> List[Dict[str, Any]]:
    """Load documents from JSON file (like your drug data)"""
    data = load_json_file(file_path)
    
    # Apply limit for testing if specified
    from ..config import config
    if config.MAX_DOCS_FOR_TESTING:
        data = data[:config.MAX_DOCS_FOR_TESTING]
        logger.info(f"Limited to {config.MAX_DOCS_FOR_TESTING} documents for testing")
    
    return data