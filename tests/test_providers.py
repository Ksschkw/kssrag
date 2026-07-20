import pytest
from unittest.mock import patch, MagicMock

from kssrag.models.base import LLMError, BaseLLM
from kssrag.models.openai_compatible import OpenAICompatibleLLM
from kssrag.models.ollama import OllamaLLM
from kssrag.models.anthropic import AnthropicLLM, _normalize_messages
from kssrag.models.providers import get_provider, list_providers
from kssrag.models.factory import create_llm


# --- provider registry -------------------------------------------------------

def test_registry_has_core_providers():
    providers = list_providers()
    for name in ["openrouter", "openai", "groq", "ollama", "anthropic", "vllm"]:
        assert name in providers


def test_get_provider_unknown_raises():
    with pytest.raises(ValueError):
        get_provider("does-not-exist")


def test_provider_kinds():
    assert get_provider("groq").kind == "openai"
    assert get_provider("anthropic").kind == "anthropic"
    assert get_provider("ollama").kind == "ollama"
    assert get_provider("vllm").requires_key is False
    assert get_provider("openai").requires_key is True


# --- factory -----------------------------------------------------------------

def test_factory_openai_compatible(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    llm = create_llm(provider="groq", model="llama-3.3-70b")
    assert isinstance(llm, OpenAICompatibleLLM)
    assert "groq.com" in llm.base_url
    assert llm.headers["Authorization"] == "Bearer gsk-test"


def test_factory_ollama_native():
    llm = create_llm(provider="ollama", model="llama3")
    assert isinstance(llm, OllamaLLM)
    assert "/api/chat" in llm.base_url


def test_factory_anthropic():
    llm = create_llm(provider="anthropic", model="claude-sonnet-4-6", api_key="sk-ant")
    assert isinstance(llm, AnthropicLLM)
    assert llm.headers["x-api-key"] == "sk-ant"


def test_factory_custom_endpoint():
    llm = create_llm(provider="custom", base_url="http://x:9000/v1/chat/completions",
                     api_key="k", model="m")
    assert isinstance(llm, OpenAICompatibleLLM)
    assert llm.base_url == "http://x:9000/v1/chat/completions"


def test_factory_keyless_local_ok():
    llm = create_llm(provider="vllm", model="mistral")
    assert isinstance(llm, OpenAICompatibleLLM)
    assert "Authorization" not in llm.headers


def test_factory_missing_required_key_raises(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)

    class _Cfg:
        PROVIDER = "mistral"
        DEFAULT_MODEL = "m"
        FALLBACK_MODELS = []
        LLM_API_KEY = None
        OPENROUTER_API_KEY = ""
        LLM_BASE_URL = ""
        LLM_TEMPERATURE = 0.7
        LLM_MAX_TOKENS = 100
        LLM_TIMEOUT = 30
        LLM_STREAM_TIMEOUT = 60

    with pytest.raises(ValueError):
        create_llm(provider="mistral", model="x", cfg=_Cfg())


# --- OpenAI-compatible adapter behavior --------------------------------------

def test_openai_compatible_predict():
    def fake_post(url, headers, json, timeout, **kw):
        m = MagicMock(); m.raise_for_status = lambda: None
        m.json = lambda: {"choices": [{"message": {"content": "hi there"}}]}
        return m
    with patch("kssrag.models.openai_compatible.requests.post", fake_post):
        llm = OpenAICompatibleLLM(base_url="http://x/v1", api_key="k", model="m")
        assert llm.predict([{"role": "user", "content": "hi"}]) == "hi there"


def test_openai_compatible_stream():
    def fake_post(url, headers, json, timeout, stream, **kw):
        m = MagicMock(); m.raise_for_status = lambda: None
        m.iter_lines = lambda: iter([
            b'data: {"choices":[{"delta":{"content":"Hel"}}]}',
            b'data: {"choices":[{"delta":{"content":"lo"}}]}',
            b'data: [DONE]',
        ])
        return m
    with patch("kssrag.models.openai_compatible.requests.post", fake_post):
        llm = OpenAICompatibleLLM(base_url="http://x/v1", api_key="k", model="m")
        assert "".join(llm.predict_stream([{"role": "user", "content": "hi"}])) == "Hello"


def test_openai_compatible_all_fail_raises():
    def fake_post(*a, **k):
        raise ConnectionError("down")
    with patch("kssrag.models.openai_compatible.requests.post", fake_post):
        llm = OpenAICompatibleLLM(base_url="http://x/v1", api_key="k", model="m", fallback_models=["b"])
        with pytest.raises(LLMError):
            llm.predict([{"role": "user", "content": "hi"}])


# --- Ollama adapter ----------------------------------------------------------

def test_ollama_predict():
    def fake_post(url, headers, json, timeout, **kw):
        m = MagicMock(); m.raise_for_status = lambda: None
        m.json = lambda: {"message": {"content": "from ollama"}, "done": True}
        return m
    with patch("kssrag.models.ollama.requests.post", fake_post):
        assert OllamaLLM(model="llama3").predict([{"role": "user", "content": "hi"}]) == "from ollama"


def test_ollama_stream_jsonl():
    def fake_post(url, headers, json, timeout, stream, **kw):
        m = MagicMock(); m.raise_for_status = lambda: None
        m.iter_lines = lambda: iter([
            b'{"message":{"content":"Hel"},"done":false}',
            b'{"message":{"content":"lo"},"done":true}',
        ])
        return m
    with patch("kssrag.models.ollama.requests.post", fake_post):
        assert "".join(OllamaLLM(model="llama3").predict_stream([{"role": "user", "content": "hi"}])) == "Hello"


# --- Anthropic adapter -------------------------------------------------------

def test_anthropic_normalize_multi_system():
    msgs = [
        {"role": "system", "content": "sysA"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "system", "content": "sysB"},
        {"role": "user", "content": "u2"},
    ]
    system, turns = _normalize_messages(msgs)
    assert "sysA" in system and "sysB" in system
    assert all(t["role"] in ("user", "assistant") for t in turns)
    assert turns[0]["role"] == "user"
    # alternation holds
    assert all(turns[i]["role"] != turns[i + 1]["role"] for i in range(len(turns) - 1))


def test_anthropic_predict():
    def fake_post(url, headers, json, timeout, **kw):
        assert headers["x-api-key"] == "sk-ant"
        m = MagicMock(); m.raise_for_status = lambda: None
        m.json = lambda: {"content": [{"type": "text", "text": "claude hi"}]}
        return m
    with patch("kssrag.models.anthropic.requests.post", fake_post):
        llm = AnthropicLLM(api_key="sk-ant", model="claude-sonnet-4-6")
        assert llm.predict([{"role": "user", "content": "hi"}]) == "claude hi"


def test_anthropic_stream():
    def fake_post(url, headers, json, timeout, stream, **kw):
        m = MagicMock(); m.raise_for_status = lambda: None
        m.iter_lines = lambda: iter([
            b'data: {"type":"content_block_delta","delta":{"text":"Cla"}}',
            b'data: {"type":"content_block_delta","delta":{"text":"ude"}}',
            b'data: {"type":"message_stop"}',
        ])
        return m
    with patch("kssrag.models.anthropic.requests.post", fake_post):
        llm = AnthropicLLM(api_key="sk-ant", model="claude-sonnet-4-6")
        assert "".join(llm.predict_stream([{"role": "user", "content": "hi"}])) == "Claude"


def test_adapters_are_base_llm():
    assert issubclass(OpenAICompatibleLLM, BaseLLM)
    assert issubclass(OllamaLLM, BaseLLM)
    assert issubclass(AnthropicLLM, BaseLLM)
