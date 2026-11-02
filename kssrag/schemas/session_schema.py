"""
Session management schemas with critical action confirmation (§13).
"""
from typing import Optional
from pydantic import BaseModel, Field, ConfigDict


class SessionClearResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str


class SessionDeleteRequestSchema(BaseModel):
    """
    Critical action confirmation schema (§13).
    Deleting a session permanently destroys conversational history.
    """
    model_config = ConfigDict(extra="forbid")
    confirmed: bool = Field(
        default=False,
        description="Explicit confirmation that this irreversible session deletion should proceed.",
    )


class SessionDeleteResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str
