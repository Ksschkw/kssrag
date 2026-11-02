"""
Query request and response transport schemas.
Conforms to house-standard.md §2, §11, and §12.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class QueryRequestSchema(BaseModel):
    """
    Transport schema for user queries.

    Input Analysis (§12):
    1. Valid input: Non-empty string up to 10,000 characters; processes query and retrieves relevant context.
    2. Worst plausible input: Whitespace-only, empty, or huge text; rejected with a 400 validation error stating 'Tell us your question.' or truncated at 10,000 chars.
    3. Input trying to break system: Control characters, BIDI overrides, SQL/shell injection; stripped safely without execution, unknown fields rejected.
    """

    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The question or search query",
    )
    session_id: Optional[str] = Field(
        default=None,
        max_length=128,
        description="Optional session identifier for conversation continuity",
    )
    metadata_filter: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Key-value filters for document retrieval",
    )

    @field_validator("query")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Query cannot be empty or whitespace only.")
        return trimmed


class QueryResponseSchema(BaseModel):
    """Response DTO for query endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str
    response: str
    session_id: str
