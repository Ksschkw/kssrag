"""
Base types shared by all LLM adapters.

Every adapter implements the same tiny interface so the rest of KSS RAG (agents,
server, CLI) is provider-agnostic:

    predict(messages)        -> str
    predict_stream(messages) -> Generator[str, None, None]

`messages` is the OpenAI-style list of {"role": ..., "content": ...} dicts.
Adapters that talk to non-OpenAI APIs (e.g. Anthropic, native Ollama) translate
this shape internally.
"""
from typing import Generator, List, Dict


class LLMError(RuntimeError):
    """Raised when an LLM adapter cannot produce a response (all models failed)."""


class BaseLLM:
    """Interface implemented by every LLM adapter."""

    def predict(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError("Subclasses must implement predict()")

    def predict_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        raise NotImplementedError("Subclasses must implement predict_stream()")
