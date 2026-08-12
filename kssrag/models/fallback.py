"""
Multi-provider fallback chain.

A single ``BaseLLM`` facade that wraps several provider adapters (e.g. Groq
primary, then OpenRouter, then Anthropic) and tries them in order until one
produces a response.

This is *cross-provider* fallback: each adapter in the chain talks to a
different provider with its own endpoint and API key. The existing
``fallback_models`` mechanism is *within-provider* — it only tries other models
on the same endpoint. The two compose: each adapter in the chain may itself
have fallback models.
"""
from typing import Generator, List, Dict, Optional

from ..utils.helpers import logger
from .base import BaseLLM, LLMError


class FallbackChainLLM(BaseLLM):
    """Try a list of adapters in order until one responds; raise LLMError if all fail."""

    def __init__(self, adapters: List[BaseLLM], names: Optional[List[str]] = None):
        if not adapters:
            raise ValueError("FallbackChainLLM requires at least one adapter")
        self.adapters = adapters
        self.names = names or [f"adapter-{i}" for i in range(len(adapters))]
        if len(self.names) != len(self.adapters):
            raise ValueError("names must have one entry per adapter")

    def _describe(self, i: int) -> str:
        return self.names[i]

    def predict(self, messages: List[Dict[str, str]]) -> str:
        """Return the first adapter's response; raise LLMError if every adapter fails."""
        errors = []
        for i, adapter in enumerate(self.adapters):
            try:
                result = adapter.predict(messages)
                logger.info(f"Provider chain: {self._describe(i)} responded")
                return result
            except LLMError as e:
                # Adapters normalize provider failures to LLMError; anything else
                # is a genuine bug and must propagate, not be masked by fallback.
                errors.append(f"{self._describe(i)}: {e}")
                logger.warning(f"Provider chain: {self._describe(i)} failed ({e}); trying next")
        raise LLMError(
            "All providers in the fallback chain failed: " + "; ".join(errors)
        )

    def predict_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """
        Stream from the first adapter that produces output.

        If an adapter fails *before* yielding anything, the next provider is
        tried. If it fails mid-stream (after yielding tokens), the error is
        re-raised — silently switching providers mid-stream would corrupt the
        response the client has already received.
        """
        errors = []
        for i, adapter in enumerate(self.adapters):
            yielded_any = False
            try:
                for chunk in adapter.predict_stream(messages):
                    yielded_any = True
                    yield chunk
                logger.info(f"Provider chain: {self._describe(i)} streamed successfully")
                return
            except LLMError as e:
                if yielded_any:
                    raise  # can't recover mid-stream
                errors.append(f"{self._describe(i)}: {e}")
                logger.warning(f"Provider chain: {self._describe(i)} failed before streaming ({e})")
        raise LLMError(
            "All providers in the fallback chain failed: " + "; ".join(errors)
        )
