"""
Health check schema.
Conforms to house-standard.md §11 (Failure behavior: liveness and readiness only).
"""
from pydantic import BaseModel, ConfigDict


class HealthResponseSchema(BaseModel):
    """Liveness probe response. Excludes version and internal configuration (§11)."""

    model_config = ConfigDict(extra="forbid")
    status: str
    message: str
