"""
Document ingestion and indexing service.
Conforms to house-standard.md §2.
"""
from typing import Dict, Any, List, Optional
from ..models.entities.document_model import DocumentModel
from ..crud.document_crud import DocumentCrud
from ..core.errors import ValidationError, NotFoundError


class DocumentService:
    """Service orchestrating document loading, chunking, and index persistence."""

    def __init__(self, document_crud: Optional[DocumentCrud] = None):
        self.document_crud = document_crud or DocumentCrud()

    def ingest_documents(self, raw_chunks: List[Dict[str, Any]]) -> List[DocumentModel]:
        """Convert raw chunks to domain DocumentModel entities and index them."""
        doc_models = []
        for chunk in raw_chunks:
            if not isinstance(chunk, dict) or "content" not in chunk:
                raise ValidationError(
                    "Invalid chunk format: expected dict with 'content'.",
                    operation="ingest_documents",
                    layer="service",
                    details={"field": "chunk", "fix": "Ensure each chunk contains a 'content' string."},
                )
            doc_models.append(
                DocumentModel(
                    content=chunk["content"],
                    metadata=chunk.get("metadata", {}),
                )
            )

        self.document_crud.set_documents(doc_models)
        return doc_models
