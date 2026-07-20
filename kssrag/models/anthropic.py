"""
Native Anthropic (Claude) adapter.

Anthropic's Messages API (/v1/messages) is NOT OpenAI-compatible:
  - auth via the `x-api-key` header (not `Authorization: Bearer`)
  - a required `anthropic-version` header
  - the system prompt is a top-level `system` field, not a message with role
    "system"
  - `messages` may only contain "user"/"assistant" roles and must start with a
    user turn and alternate roles
  - responses look like {"content": [{"type": "text", "text": "..."}]}
  - streaming is SSE with typed events; text arrives in `content_block_delta`
    events as `delta.text`

Because KSS RAG's agent emits OpenAI-style messages (including *multiple* system
messages for summaries and the stealth-summary instruction), this adapter
normalizes them: all system messages are concatenated into the top-level
`system` field, and consecutive same-role turns are merged so the alternation
requirement holds.
"""
import json
from typing import Generator, List, Dict, Optional, Tuple

import requests

from ..utils.helpers import logger
from ..config import config
from .base import BaseLLM, LLMError

DEFAULT_ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"


def _normalize_messages(messages: List[Dict[str, str]]) -> Tuple[str, List[Dict[str, str]]]:
    """Split OpenAI-style messages into (system_text, alternating_turns)."""
    system_parts: List[str] = []
    turns: List[Dict[str, str]] = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            if content:
                system_parts.append(content)
            continue
        # Map any non-system role to user/assistant.
        role = "assistant" if role == "assistant" else "user"
        if turns and turns[-1]["role"] == role:
            # Merge consecutive same-role turns to preserve alternation.
            turns[-1]["content"] += "\n\n" + content
        else:
            turns.append({"role": role, "content": content})

    # Anthropic requires the first turn to be from the user.
    if turns and turns[0]["role"] == "assistant":
        turns.insert(0, {"role": "user", "content": "(continued)"})

    return "\n\n".join(system_parts), turns


class AnthropicLLM(BaseLLM):
    """Adapter for Anthropic's native Messages API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        stream_timeout: Optional[int] = None,
    ):
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.base_url = base_url or DEFAULT_ANTHROPIC_URL
        self.model = model or config.DEFAULT_MODEL
        self.fallback_models = fallback_models if fallback_models is not None else config.FALLBACK_MODELS
        self.stream = stream
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS
        self.timeout = timeout if timeout is not None else config.LLM_TIMEOUT
        self.stream_timeout = stream_timeout if stream_timeout is not None else config.LLM_STREAM_TIMEOUT
        self.headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key or "",
            "anthropic-version": ANTHROPIC_VERSION,
        }

    @property
    def _models_to_try(self) -> List[str]:
        return [self.model] + list(self.fallback_models or [])

    def _payload(self, model: str, system: str, turns: List[Dict[str, str]], stream: bool) -> Dict:
        payload = {
            "model": model,
            "messages": turns,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "stream": stream,
        }
        if system:
            payload["system"] = system
        return payload

    def predict(self, messages: List[Dict[str, str]]) -> str:
        if self.stream:
            return "".join(self.predict_stream(messages))

        system, turns = _normalize_messages(messages)
        logger.info(f"Generating Anthropic response ({len(turns)} turns)")
        for model in self._models_to_try:
            try:
                logger.info(f"Using Anthropic model: {model}")
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=self._payload(model, system, turns, stream=False),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                blocks = data.get("content") or []
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                if not text:
                    logger.warning(f"Empty Anthropic response from {model}: {data}")
                    continue
                return text
            except Exception as e:
                logger.warning(f"Anthropic request failed with model {model}: {str(e)}")
                continue

        logger.error("All Anthropic models failed to respond")
        raise LLMError("Unable to generate response from Anthropic.")

    def predict_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        system, turns = _normalize_messages(messages)
        logger.info(f"Streaming Anthropic response ({len(turns)} turns)")
        for model in self._models_to_try:
            try:
                logger.info(f"Streaming with Anthropic model: {model}")
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=self._payload(model, system, turns, stream=True),
                    timeout=self.stream_timeout,
                    stream=True,
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:].strip()
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    etype = event.get("type")
                    if etype == "content_block_delta":
                        delta = event.get("delta") or {}
                        text = delta.get("text")
                        if text:
                            yield text
                    elif etype == "message_stop":
                        logger.info("Anthropic stream completed successfully")
                        return
                return
            except Exception as e:
                logger.warning(f"Anthropic streaming failed with model {model}: {str(e)}")
                continue

        logger.error("All Anthropic models failed for streaming")
        raise LLMError("Unable to stream response from Anthropic.")
