"""
Provider preset registry.

Maps a short provider name to everything needed to build its adapter: the chat
endpoint URL, which environment variable holds its API key, whether a key is
required, and which adapter kind speaks its protocol.

Most providers share the OpenAI /chat/completions wire format, so they use the
"openai" adapter kind and differ only by base_url + key env var. Providers with
their own protocols (Anthropic Messages API, native Ollama) declare a different
kind, handled by the factory.
"""
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class ProviderPreset:
    name: str
    base_url: str
    api_key_env: Optional[str]   # env var holding the key (None for keyless local servers)
    requires_key: bool
    kind: str = "openai"         # "openai" | "anthropic" | "ollama"


# Hosted, OpenAI-compatible providers -------------------------------------------------
_OPENAI_COMPATIBLE = {
    "openrouter":  ("https://openrouter.ai/api/v1/chat/completions", "OPENROUTER_API_KEY"),
    "openai":      ("https://api.openai.com/v1/chat/completions", "OPENAI_API_KEY"),
    "groq":        ("https://api.groq.com/openai/v1/chat/completions", "GROQ_API_KEY"),
    "together":    ("https://api.together.xyz/v1/chat/completions", "TOGETHER_API_KEY"),
    "deepseek":    ("https://api.deepseek.com/v1/chat/completions", "DEEPSEEK_API_KEY"),
    "fireworks":   ("https://api.fireworks.ai/inference/v1/chat/completions", "FIREWORKS_API_KEY"),
    "mistral":     ("https://api.mistral.ai/v1/chat/completions", "MISTRAL_API_KEY"),
    "perplexity":  ("https://api.perplexity.ai/chat/completions", "PERPLEXITY_API_KEY"),
    "xai":         ("https://api.x.ai/v1/chat/completions", "XAI_API_KEY"),
    "deepinfra":   ("https://api.deepinfra.com/v1/openai/chat/completions", "DEEPINFRA_API_KEY"),
    "anyscale":    ("https://api.endpoints.anyscale.com/v1/chat/completions", "ANYSCALE_API_KEY"),
}

# Local, OpenAI-compatible servers (no API key required) ------------------------------
_LOCAL_OPENAI = {
    "ollama-openai": "http://localhost:11434/v1/chat/completions",
    "lmstudio":      "http://localhost:1234/v1/chat/completions",
    "vllm":          "http://localhost:8000/v1/chat/completions",
    "llamacpp":      "http://localhost:8080/v1/chat/completions",
}

PROVIDERS: Dict[str, ProviderPreset] = {}

for _name, (_url, _env) in _OPENAI_COMPATIBLE.items():
    PROVIDERS[_name] = ProviderPreset(_name, _url, _env, requires_key=True, kind="openai")

for _name, _url in _LOCAL_OPENAI.items():
    PROVIDERS[_name] = ProviderPreset(_name, _url, None, requires_key=False, kind="openai")

# Native (non-OpenAI) protocols -------------------------------------------------------
PROVIDERS["anthropic"] = ProviderPreset(
    "anthropic", "https://api.anthropic.com/v1/messages",
    "ANTHROPIC_API_KEY", requires_key=True, kind="anthropic",
)
PROVIDERS["ollama"] = ProviderPreset(
    "ollama", "http://localhost:11434/api/chat",
    None, requires_key=False, kind="ollama",
)


def get_provider(name: str) -> ProviderPreset:
    """Look up a provider preset by name (case-insensitive)."""
    key = (name or "").strip().lower()
    if key not in PROVIDERS:
        available = ", ".join(sorted(PROVIDERS))
        raise ValueError(f"Unknown provider '{name}'. Available: {available}")
    return PROVIDERS[key]


def list_providers() -> Dict[str, ProviderPreset]:
    """Return all registered provider presets."""
    return dict(PROVIDERS)
