"""
Local / offline LLM helpers.
"""
from typing import Generator, List, Dict

from ..utils.helpers import logger
from .base import BaseLLM


class MockLLM(BaseLLM):
    """Deterministic mock LLM for tests and offline development (no network)."""

    def _last_user_message(self, messages: List[Dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg.get("role") == "user":
                return msg.get("content", "")
        return ""

    def predict(self, messages: List[Dict[str, str]]) -> str:
        logger.info("Using mock LLM for response generation")
        return f"This is a mock response to: {self._last_user_message(messages)}"

    def predict_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        for token in self.predict(messages).split(" "):
            yield token + " "
