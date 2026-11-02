"""
Query and result domain entities.
Conforms to house-standard.md §2.
"""
from dataclasses import dataclass, field
import time
from typing import List, Dict, Any, Optional
from .document_model import DocumentModel


@dataclass(frozen=True)
class QueryResultModel:
    """Domain model representing the outcome of a RAG query execution."""

    query: str
    response: str
    session_id: str
    retrieved_documents: List[DocumentModel] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
