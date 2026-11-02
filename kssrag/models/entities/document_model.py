"""
Document domain entity.
Conforms to house-standard.md §2.
"""
from dataclasses import dataclass, field
from typing import Dict, Any


@dataclass(frozen=True)
class DocumentModel:
    """Domain model representing an indexed document chunk with invariants."""

    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.content, str):
            raise TypeError("DocumentModel content must be a string")
        if not isinstance(self.metadata, dict):
            raise TypeError("DocumentModel metadata must be a dictionary")

    @property
    def identifier(self) -> Any:
        """Return stable identifier from metadata if present."""
        return self.metadata.get("id")

    @property
    def source(self) -> str:
        """Return source filepath or identifier."""
        return str(self.metadata.get("source", ""))
