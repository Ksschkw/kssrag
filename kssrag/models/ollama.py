"""
Native Ollama adapter.

Talks to Ollama's own /api/chat endpoint (not the OpenAI-compat shim at /v1).
Use this for a vanilla local Ollama install. Ollama streams newline-delimited
JSON objects, each shaped like {"message": {"content": "..."}, "done": bool} —
not Server-Sent Events, so it is parsed differently from the OpenAI adapter.

No API key is required. Ollama has no server-side fallback-model concept, but we
still honor the configured fallback list by retrying with the next model if one
fails (e.g. model not pulled locally).
"""
import json
from typing import Generator, List, Dict, Optional

import requests

from ..utils.helpers import logger
from ..config import config
from .base import BaseLLM, LLMError

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"


class OllamaLLM(BaseLLM):
    """Adapter for a local Ollama server via its native /api/chat endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        stream_timeout: Optional[int] = None,
    ):
        self.base_url = base_url or DEFAULT_OLLAMA_URL
        self.model = model or config.DEFAULT_MODEL
        self.fallback_models = fallback_models if fallback_models is not None else config.FALLBACK_MODELS
        self.stream = stream
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS
        self.timeout = timeout if timeout is not None else config.LLM_TIMEOUT
        self.stream_timeout = stream_timeout if stream_timeout is not None else config.LLM_STREAM_TIMEOUT
        self.headers = {"Content-Type": "application/json"}

    @property
    def _models_to_try(self) -> List[str]:
        return [self.model] + list(self.fallback_models or [])

    def _payload(self, model: str, messages: List[Dict[str, str]], stream: bool) -> Dict:
        return {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

    def predict(self, messages: List[Dict[str, str]]) -> str:
        if self.stream:
            return "".join(self.predict_stream(messages))

        logger.info(f"Generating Ollama response with {len(messages)} messages")
        for model in self._models_to_try:
            try:
                logger.info(f"Using Ollama model: {model}")
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=self._payload(model, messages, stream=False),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                data = response.json()
                content = (data.get("message") or {}).get("content")
                if not content:
                    logger.warning(f"Empty Ollama response from {model}: {data}")
                    continue
                return content
            except Exception as e:
                logger.warning(f"Ollama request failed with model {model}: {str(e)}")
                continue

        logger.error("All Ollama models failed to respond")
        raise LLMError("Unable to generate response from Ollama.")

    def predict_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        logger.info(f"Streaming Ollama response with {len(messages)} messages")
        for model in self._models_to_try:
            try:
                logger.info(f"Streaming with Ollama model: {model}")
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=self._payload(model, messages, stream=True),
                    timeout=self.stream_timeout,
                    stream=True,
                )
                response.raise_for_status()

                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse Ollama chunk: {str(e)}")
                        continue
                    content = (chunk.get("message") or {}).get("content")
                    if content:
                        yield content
                    if chunk.get("done"):
                        logger.info("Ollama stream completed successfully")
                        return
                return
            except Exception as e:
                logger.warning(f"Ollama streaming failed with model {model}: {str(e)}")
                continue

        logger.error("All Ollama models failed for streaming")
        raise LLMError("Unable to stream response from Ollama.")
