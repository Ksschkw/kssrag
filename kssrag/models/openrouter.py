"""
OpenRouter LLM adapter.

Thin preset over OpenAICompatibleLLM pinned to OpenRouter's endpoint. Kept for
backward compatibility; new code should prefer the provider factory
(kssrag.models.factory.create_llm) or OpenAICompatibleLLM directly.
"""
from typing import List, Optional

from ..config import config
from .base import LLMError
from .openai_compatible import OpenAICompatibleLLM

__all__ = ["OpenRouterLLM", "LLMError"]

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterLLM(OpenAICompatibleLLM):
    """OpenAI-compatible adapter pinned to OpenRouter, with OpenRouter's headers."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None,
                 fallback_models: Optional[List[str]] = None, stream: bool = False,
                 temperature: Optional[float] = None, max_tokens: Optional[int] = None,
                 timeout: Optional[int] = None, stream_timeout: Optional[int] = None):
        super().__init__(
            base_url=OPENROUTER_BASE_URL,
            api_key=api_key if api_key is not None else config.OPENROUTER_API_KEY,
            model=model,
            fallback_models=fallback_models,
            stream=stream,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
            stream_timeout=stream_timeout,
            extra_headers={
                "HTTP-Referer": "https://github.com/Ksschkw/kssrag",
                "X-Title": "KSSRAG",
            },
        )
