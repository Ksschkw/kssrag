"""
LLM factory: build the right adapter from a provider name plus overrides.

Resolution order (each step overridable by the next-more-specific source):
  1. provider preset  -> base_url, api-key env var, adapter kind
  2. environment/config -> LLM_BASE_URL / LLM_API_KEY / DEFAULT_MODEL
  3. explicit kwargs   -> whatever the caller (e.g. a CLI flag) passes

This keeps the common case trivial (`create_llm()` uses config defaults) while
allowing full control (`create_llm(provider="groq", model="llama-3.3-70b")`).

Cross-provider fallback: pass `fallback_providers` (or set `FALLBACK_PROVIDERS`
in config) to wrap the primary adapter in a chain that tries other providers in
order when the primary fails — e.g. Groq first, OpenRouter as backup. Each
entry may be a provider name, a (provider, kwargs) tuple, or a kwargs dict that
includes a "provider" key.
"""
import os
from typing import Dict, List, Optional, Tuple, Union

from ..config import config as default_config
from ..utils.helpers import logger
from .base import BaseLLM
from .providers import get_provider
from .openai_compatible import OpenAICompatibleLLM
from .ollama import OllamaLLM
from .anthropic import AnthropicLLM
from .fallback import FallbackChainLLM

FallbackEntry = Union[str, Tuple[str, Dict], Dict]


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


def _build_adapter(
    provider: str,
    model: Optional[str],
    base_url: Optional[str],
    api_key: Optional[str],
    fallback_models: Optional[List[str]],
    stream: bool,
    cfg,
    kwargs: Dict,
) -> BaseLLM:
    """Construct a single adapter for one provider (primary or fallback)."""
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


def _entry_to_kwargs(entry: FallbackEntry, index: int) -> Dict:
    """Normalize a fallback entry to a kwargs dict with a 'provider' key."""
    if isinstance(entry, str):
        return {"provider": entry}
    if isinstance(entry, (tuple, list)):
        if len(entry) != 2 or not isinstance(entry[1], dict):
            raise ValueError(
                f"Fallback provider #{index} must be a name, (name, kwargs) tuple, or dict"
            )
        return {"provider": entry[0], **entry[1]}
    if isinstance(entry, dict):
        if "provider" not in entry:
            raise ValueError(f"Fallback provider #{index} dict must include 'provider'")
        return dict(entry)
    raise ValueError(f"Invalid fallback provider #{index}: {entry!r}")


def create_llm(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    fallback_models: Optional[List[str]] = None,
    fallback_providers: Optional[List[FallbackEntry]] = None,
    stream: bool = False,
    cfg=None,
    **kwargs,
) -> BaseLLM:
    """
    Construct an LLM adapter for the given provider, optionally wrapped in a
    cross-provider fallback chain.

    Args:
        provider: preset name (defaults to cfg.PROVIDER). Use "custom" to
            require an explicit base_url with the OpenAI-compatible adapter.
        model: model id (defaults to cfg.DEFAULT_MODEL via each adapter).
        base_url: overrides the provider's endpoint.
        api_key: overrides the resolved API key.
        fallback_models: same-provider model fallbacks (forwarded to the adapter).
        fallback_providers: cross-provider fallback chain tried in order if the
            primary fails. Each entry is a provider name (uses global defaults),
            a (provider, kwargs) tuple, or a dict that includes "provider"
            (plus optional model/base_url/api_key/fallback_models overrides).
            Defaults to cfg.FALLBACK_PROVIDERS when unset.
        cfg: Config instance to read defaults from (defaults to the global config).
        stream, **kwargs: forwarded to every adapter in the chain.
    """
    cfg = cfg if cfg is not None else default_config
    provider = (provider or cfg.PROVIDER or "openrouter").strip().lower()

    if fallback_providers is None:
        fallback_providers = list(getattr(cfg, "FALLBACK_PROVIDERS", None) or [])

    primary = _build_adapter(
        provider, model, base_url, api_key, fallback_models, stream, cfg, kwargs,
    )

    if not fallback_providers:
        return primary

    chain = [primary]
    names = [f"{provider}:{model or 'default'}"]

    for index, entry in enumerate(fallback_providers, start=1):
        entry_kwargs = _entry_to_kwargs(entry, index)
        f_provider = entry_kwargs.pop("provider").strip().lower()
        if "model" not in entry_kwargs:
            logger.warning(
                f"Fallback provider '{f_provider}' has no explicit model; it will use "
                f"DEFAULT_MODEL ({cfg.DEFAULT_MODEL}), which may not be valid for that "
                f"provider. Pass a per-provider model via fallback_providers."
            )
        try:
            adapter = _build_adapter(
                f_provider,
                entry_kwargs.get("model"),
                entry_kwargs.get("base_url"),
                entry_kwargs.get("api_key"),
                entry_kwargs.get("fallback_models"),
                stream,
                cfg,
                kwargs,
            )
        except ValueError as e:
            # A fallback provider that isn't configured (e.g. missing key)
            # shouldn't break the whole chain — skip it with a warning.
            logger.warning(f"Skipping fallback provider '{f_provider}': {e}")
            continue
        chain.append(adapter)
        names.append(f"{f_provider}:{entry_kwargs.get('model') or 'default'}")

    if len(chain) == 1:
        return primary
    return FallbackChainLLM(chain, names)
