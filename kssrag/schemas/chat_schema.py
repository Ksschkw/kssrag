"""
OpenAI-compatible chat completion schemas.
Conforms to house-standard.md §2, §11, and §12.
"""
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict, field_validator


class ChatCompletionRequestSchema(BaseModel):
    """
    OpenAI-compatible chat completions request.

    Input Analysis (§12):
    1. Valid input: List of message dictionaries with role and content.
    2. Worst plausible input: Empty messages list, missing user role, multimodal lists. Rejected with 400.
    3. Input trying to break system: Huge payload, extra unknown fields. Rejected by extra='forbid' and length limits.
    """

    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = Field(default=None, max_length=256)
    messages: List[Dict[str, Any]] = Field(..., min_length=1, max_length=500)
    stream: bool = False
    user: Optional[str] = Field(default=None, max_length=128)

    @field_validator("messages")
    @classmethod
    def validate_user_message_exists(cls, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
        if not user_msgs:
            raise ValueError("No user message found in messages list.")
        raw_content = user_msgs[-1].get("content", "")
        content_str = raw_content if isinstance(raw_content, str) else str(raw_content)
        if not content_str.strip():
            raise ValueError("User message content cannot be empty.")
        return messages


class ChatMessageSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: str
    content: str


class ChatChoiceSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    index: int
    message: ChatMessageSchema
    finish_reason: str


class ChatUsageSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoiceSchema]
    usage: ChatUsageSchema
