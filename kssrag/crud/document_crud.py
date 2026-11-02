"""
Document persistence CRUD.
Conforms to house-standard.md §2.
"""
from typing import List, Optional
from ..models.entities.document_model import DocumentModel


class DocumentCrud:
    """Document and index persistence wrapper."""

    def __init__(self, vector_store=None):
        self.vector_store = vector_store
        self._documents: List[DocumentModel] = []

    def create(self, doc: DocumentModel) -> DocumentModel:
        """Add a single document entity."""
        self._documents.append(doc)
        if self.vector_store is not None:
            self.vector_store.add_documents([{"content": doc.content, "metadata": doc.metadata}])
        return doc

    def list(self) -> List[DocumentModel]:
        """List all indexed document entities."""
        return list(self._documents)

    def set_documents(self, docs: List[DocumentModel]) -> None:
        """Replace all stored documents."""
        self._documents = list(docs)
        if self.vector_store is not None:
            raw_docs = [{"content": d.content, "metadata": d.metadata} for d in docs]
            self.vector_store.add_documents(raw_docs)
