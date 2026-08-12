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
import time
from typing import Generator, List, Dict

import requests

from ..utils.helpers import logger


def _retry_after_seconds(resp, backoff: float, attempt: int, cap: float = 60.0) -> float:
    """How long to wait before retrying a 429, honoring server hints when present."""
    # HTTP standard: Retry-After as seconds.
    retry_after = resp.headers.get("Retry-After")
    if retry_after and retry_after.isdigit():
        return min(float(retry_after), cap)
    # Common LLM APIs send the epoch time the token bucket resets.
    reset = resp.headers.get("x-ratelimit-reset-tokens") or resp.headers.get("x-ratelimit-reset")
    if reset and reset.isdigit():
        return min(max(float(reset) - time.time(), 0.0), cap)
    # Exponential backoff as a fallback.
    return min(backoff * (2 ** attempt), cap)


def post_with_retry(url, *, headers, json, timeout, max_retries: int, backoff: float, stream: bool = False):
    """
    POST to a chat endpoint, retrying the *same* request on 429 rate limits.

    On a 429, waits for the server's Retry-After / rate-limit-reset hint (or an
    exponential backoff) and retries up to ``max_retries`` times. Only when the
    retries are exhausted does the 429 surface as a RequestException, letting the
    caller move on to the next fallback model/provider.

    Uses ``requests.post`` directly so test patches on ``requests.post`` keep
    working across modules.
    """
    for attempt in range(max_retries + 1):
        resp = requests.post(url, headers=headers, json=json, timeout=timeout, stream=stream)
        if resp.status_code == 429 and attempt < max_retries:
            wait = _retry_after_seconds(resp, backoff, attempt)
            logger.warning(
                f"Rate limited (429) by {url}; retrying in {wait:.1f}s "
                f"(attempt {attempt + 1}/{max_retries})"
            )
            time.sleep(wait)
            continue
        break
    resp.raise_for_status()
    return resp


class LLMError(RuntimeError):
    """Raised when an LLM adapter cannot produce a response (all models failed)."""


class BaseLLM:
    """Interface implemented by every LLM adapter."""

    def predict(self, messages: List[Dict[str, str]]) -> str:
        raise NotImplementedError("Subclasses must implement predict()")

    def predict_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        raise NotImplementedError("Subclasses must implement predict_stream()")
