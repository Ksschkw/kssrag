"""
Session domain entity.
Conforms to house-standard.md §2.
"""
from dataclasses import dataclass, field
import time
from typing import List


@dataclass(frozen=True)
class ConversationMessageModel:
    """A single message within a conversation."""

    role: str
    content: str
    created_at: float = field(default_factory=time.time)

    def __post_init__(self):
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(f"Invalid message role '{self.role}'. Expected system, user, or assistant.")
        if not isinstance(self.content, str):
            raise TypeError("ConversationMessageModel content must be a string")


@dataclass
class SessionModel:
    """Domain model representing a multi-turn conversation session."""

    session_id: str
    messages: List[ConversationMessageModel] = field(default_factory=list)
    summaries: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def add_message(self, role: str, content: str) -> None:
        """Add a message and update timestamp."""
        msg = ConversationMessageModel(role=role, content=content)
        self.messages.append(msg)
        self.updated_at = time.time()

    def clear(self) -> None:
        """Clear conversation messages while keeping session record."""
        self.messages.clear()
        self.summaries.clear()
        self.updated_at = time.time()
