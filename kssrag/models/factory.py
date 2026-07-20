"""
LLM factory: build the right adapter from a provider name plus overrides.

Resolution order (each step overridable by the next-more-specific source):
  1. provider preset  -> base_url, api-key env var, adapter kind
  2. environment/config -> LLM_BASE_URL / LLM_API_KEY / DEFAULT_MODEL
  3. explicit kwargs   -> whatever the caller (e.g. a CLI flag) passes

This keeps the common case trivial (`create_llm()` uses config defaults) while
allowing full control (`create_llm(provider="groq", model="llama-3.3-70b")`).
"""
import os
from typing import List, Optional

from ..config import config as default_config
from .base import BaseLLM
from .providers import get_provider
from .openai_compatible import OpenAICompatibleLLM
from .ollama import OllamaLLM
from .anthropic import AnthropicLLM


def _resolve_api_key(preset, explicit_key: Optional[str], cfg) -> Optional[str]:
    """Pick the API key: explicit > LLM_API_KEY config > provider env var > OpenRouter key."""
    if explicit_key:
        return explicit_key
    if cfg.LLM_API_KEY:
        return cfg.LLM_API_KEY
    if preset.api_key_env:
        env_val = os.getenv(preset.api_key_env)
        if env_val:
            return env_val
    # Legacy fallback so existing OPENROUTER_API_KEY setups keep working.
    return cfg.OPENROUTER_API_KEY or None


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
    stream: bool = False,
    cfg=None,
    **kwargs,
) -> BaseLLM:
    """
    Construct an LLM adapter for the given provider.

    Args:
        provider: preset name (defaults to cfg.PROVIDER). Use "custom" to
            require an explicit base_url with the OpenAI-compatible adapter.
        model: model id (defaults to cfg.DEFAULT_MODEL via each adapter).
        base_url: overrides the provider's endpoint.
        api_key: overrides the resolved API key.
        cfg: Config instance to read defaults from (defaults to the global config).
        fallback_models, stream, **kwargs: forwarded to the adapter.
    """
    cfg = cfg if cfg is not None else default_config
    provider = (provider or cfg.PROVIDER or "openrouter").strip().lower()

    if provider == "custom":
        url = base_url or cfg.LLM_BASE_URL
        return OpenAICompatibleLLM(
            base_url=url, api_key=api_key, model=model,
            fallback_models=fallback_models, stream=stream, **kwargs,
        )

    preset = get_provider(provider)
    url = base_url or preset.base_url
    key = _resolve_api_key(preset, api_key, cfg)

    if preset.requires_key and not key:
        raise ValueError(
            f"Provider '{provider}' requires an API key. Set {preset.api_key_env} "
            f"(or LLM_API_KEY), or pass api_key explicitly."
        )

    if preset.kind == "anthropic":
        return AnthropicLLM(
            api_key=key, base_url=url, model=model,
            fallback_models=fallback_models, stream=stream, **kwargs,
        )
    if preset.kind == "ollama":
        return OllamaLLM(
            base_url=url, model=model,
            fallback_models=fallback_models, stream=stream, **kwargs,
        )
    # default: OpenAI-compatible
    return OpenAICompatibleLLM(
        base_url=url, api_key=key, model=model,
        fallback_models=fallback_models, stream=stream, **kwargs,
    )
