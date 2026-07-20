"""
OpenAI-compatible chat LLM adapter.

Most hosted LLM providers speak the OpenAI /chat/completions wire format:
OpenRouter, OpenAI, Groq, Together, DeepSeek, Fireworks, Mistral, Perplexity,
xAI, and local servers like Ollama (OpenAI-compat mode), LM Studio, and vLLM.
This single adapter targets all of them — only the base_url and api_key differ.

The adapter tries the primary model then each fallback in turn, and raises
LLMError only if every model fails.
"""
import json
from typing import Generator, List, Dict, Optional

import requests

from ..utils.helpers import logger
from ..config import config
from .base import BaseLLM, LLMError


class OpenAICompatibleLLM(BaseLLM):
    """Chat adapter for any OpenAI /chat/completions-compatible endpoint."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        stream: bool = False,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        timeout: Optional[int] = None,
        stream_timeout: Optional[int] = None,
        extra_headers: Optional[Dict[str, str]] = None,
    ):
        self.base_url = base_url or config.LLM_BASE_URL
        self.api_key = api_key if api_key is not None else config.OPENROUTER_API_KEY
        self.model = model or config.DEFAULT_MODEL
        self.fallback_models = fallback_models if fallback_models is not None else config.FALLBACK_MODELS
        self.stream = stream
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.max_tokens = max_tokens if max_tokens is not None else config.LLM_MAX_TOKENS
        self.timeout = timeout if timeout is not None else config.LLM_TIMEOUT
        self.stream_timeout = stream_timeout if stream_timeout is not None else config.LLM_STREAM_TIMEOUT

        self.headers = {"Content-Type": "application/json"}
        # Only send Authorization when a key is present; local servers (Ollama,
        # LM Studio, vLLM) accept requests with no key.
        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"
        if extra_headers:
            self.headers.update(extra_headers)

    @property
    def _models_to_try(self) -> List[str]:
        return [self.model] + list(self.fallback_models or [])

    def _payload(self, model: str, messages: List[Dict[str, str]], stream: bool) -> Dict:
        return {
            "model": model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": 1,
            "stop": None,
            "stream": stream,
        }

    def predict(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response, trying fallback models in order."""
        if self.stream:
            return "".join(self.predict_stream(messages))

        logger.info(f"Generating response with {len(messages)} messages")

        for model in self._models_to_try:
            try:
                logger.info(f"Using model: {model}")
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=self._payload(model, messages, stream=False),
                    timeout=self.timeout,
                )
                response.raise_for_status()
                response_data = response.json()

                if ("choices" not in response_data or
                        len(response_data["choices"]) == 0 or
                        "message" not in response_data["choices"][0] or
                        "content" not in response_data["choices"][0]["message"]):
                    logger.warning(f"Invalid response format from {model}: {response_data}")
                    continue

                content = response_data["choices"][0]["message"]["content"]
                logger.info(f"Successfully generated response with model: {model}")
                return content

            except requests.exceptions.Timeout:
                logger.warning(f"Model {model} timed out")
                continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error with model {model}: {str(e)}")
                if getattr(e, "response", None) is not None:
                    try:
                        logger.warning(f"Error response: {e.response.json()}")
                    except Exception:
                        logger.warning(f"Error response text: {e.response.text}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error with model {model}: {str(e)}")
                continue

        logger.error("All model fallbacks failed to respond")
        raise LLMError("Unable to generate response from available models.")

    def predict_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Stream a response token-by-token, trying fallback models in order."""
        logger.info(f"Streaming response with {len(messages)} messages")

        for model in self._models_to_try:
            try:
                logger.info(f"Streaming with model: {model}")
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
                    line = line.decode("utf-8")
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data.strip() == "[DONE]":
                        logger.info("Stream completed successfully")
                        return
                    try:
                        chunk_data = json.loads(data)
                        choices = chunk_data.get("choices") or []
                        if choices:
                            delta = choices[0].get("delta") or {}
                            content = delta.get("content")
                            if content:
                                yield content
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse stream chunk: {str(e)}")
                        continue

                logger.info(f"Successfully streamed from model: {model}")
                return

            except Exception as e:
                logger.warning(f"Streaming failed with model {model}: {str(e)}")
                continue

        logger.error("All model fallbacks failed for streaming")
        raise LLMError("Unable to stream response from available models.")
