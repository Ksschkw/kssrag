import pytest
from unittest.mock import patch

import requests

from kssrag.models.base import BaseLLM, LLMError
from kssrag.models.fallback import FallbackChainLLM
from kssrag.models.openai_compatible import OpenAICompatibleLLM
from kssrag.models.factory import create_llm
from kssrag.core.retrievers import HybridRetriever
from kssrag.config import Config


# --- stub adapters -----------------------------------------------------------

class _Stub(BaseLLM):
    def __init__(self, text="ok", fail=False):
        self.text = text
        self.fail = fail

    def predict(self, messages):
        if self.fail:
            raise LLMError("boom")
        return self.text

    def predict_stream(self, messages):
        if self.fail:
            raise LLMError("boom")
        yield self.text


class _MidStreamStub(BaseLLM):
    def predict(self, messages):
        raise LLMError("never used")

    def predict_stream(self, messages):
        yield "partial"
        raise LLMError("mid-stream failure")


# --- FallbackChainLLM --------------------------------------------------------

def test_chain_predict_falls_back_to_second_adapter():
    chain = FallbackChainLLM([_Stub(fail=True), _Stub("backup")], ["primary", "backup"])
    assert chain.predict([{"role": "user", "content": "hi"}]) == "backup"


def test_chain_predict_uses_first_success():
    chain = FallbackChainLLM([_Stub("first"), _Stub("second")], ["a", "b"])
    assert chain.predict([{"role": "user", "content": "hi"}]) == "first"


def test_chain_stream_falls_back_before_yielding():
    chain = FallbackChainLLM([_Stub(fail=True), _Stub("backup-stream")], ["a", "b"])
    assert "".join(chain.predict_stream([{"role": "user", "content": "hi"}])) == "backup-stream"


def test_chain_stream_reraises_after_partial_output():
    # Once tokens have been emitted we must not silently switch providers.
    chain = FallbackChainLLM([_MidStreamStub(), _Stub("backup")], ["a", "b"])
    with pytest.raises(LLMError):
        list(chain.predict_stream([{"role": "user", "content": "hi"}]))


def test_chain_all_fail_raises_llm_error():
    chain = FallbackChainLLM([_Stub(fail=True), _Stub(fail=True)], ["a", "b"])
    with pytest.raises(LLMError):
        chain.predict([{"role": "user", "content": "hi"}])
    with pytest.raises(LLMError):
        list(chain.predict_stream([{"role": "user", "content": "hi"}]))


def test_chain_requires_adapter():
    with pytest.raises(ValueError):
        FallbackChainLLM([])


# --- factory chain building --------------------------------------------------

def test_factory_returns_plain_adapter_without_fallbacks(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    llm = create_llm(provider="groq", model="m", cfg=Config(FALLBACK_PROVIDERS=[]))
    assert isinstance(llm, OpenAICompatibleLLM)


def test_factory_builds_chain_from_cfg(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant")
    cfg = Config(FALLBACK_PROVIDERS=["anthropic"])
    llm = create_llm(provider="groq", model="m", cfg=cfg)
    assert isinstance(llm, FallbackChainLLM)
    assert len(llm.adapters) == 2


def test_factory_builds_chain_from_explicit_param(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    llm = create_llm(
        provider="groq",
        model="m",
        cfg=Config(FALLBACK_PROVIDERS=[]),
        fallback_providers=[("anthropic", {"api_key": "sk-ant", "model": "claude-x"})],
    )
    assert isinstance(llm, FallbackChainLLM)
    assert len(llm.adapters) == 2
    assert llm.names[1] == "anthropic:claude-x"


def test_factory_skips_unconfigured_fallback(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # openai requires a key; with none configured it must be skipped, not fatal.
    # Use an explicit cfg so the key resolver can't fall back to a machine env key.
    cfg = Config(FALLBACK_PROVIDERS=[], OPENROUTER_API_KEY="")
    llm = create_llm(
        provider="groq",
        model="m",
        cfg=cfg,
        fallback_providers=["openai"],
    )
    assert isinstance(llm, OpenAICompatibleLLM)
    assert not isinstance(llm, FallbackChainLLM)


def test_config_parses_fallback_providers_env(monkeypatch):
    monkeypatch.setenv("FALLBACK_PROVIDERS", "ollama, anthropic")
    assert Config().FALLBACK_PROVIDERS == ["ollama", "anthropic"]
    monkeypatch.delenv("FALLBACK_PROVIDERS", raising=False)


# --- 429 rate-limit retry ----------------------------------------------------

def _fake_response(status, body=None):
    class Resp:
        status_code = status
        headers = {}

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(f"{self.status_code}")

        def json(self):
            return body

    return Resp()


def test_429_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return _fake_response(429)
        return _fake_response(200, {"choices": [{"message": {"content": "recovered"}}]})

    monkeypatch.setattr("kssrag.models.base.time.sleep", lambda s: None)
    with patch("kssrag.models.openai_compatible.requests.post", fake_post):
        llm = OpenAICompatibleLLM(base_url="http://x/v1", api_key="k", model="m",
                                  fallback_models=[], max_retries=3, retry_backoff=0.1)
        assert llm.predict([{"role": "user", "content": "hi"}]) == "recovered"
    assert calls["n"] == 2


def test_429_exhausted_raises(monkeypatch):
    calls = {"n": 0}

    def fake_post(*a, **k):
        calls["n"] += 1
        return _fake_response(429)

    monkeypatch.setattr("kssrag.models.base.time.sleep", lambda s: None)
    with patch("kssrag.models.openai_compatible.requests.post", fake_post):
        llm = OpenAICompatibleLLM(base_url="http://x/v1", api_key="k", model="m",
                                  fallback_models=[], max_retries=1, retry_backoff=0.1)
        with pytest.raises(LLMError):
            llm.predict([{"role": "user", "content": "hi"}])
    assert calls["n"] == 2  # initial + 1 retry


# --- FUZZY_MATCH_THRESHOLD wiring -------------------------------------------

class _Store:
    def retrieve(self, query, top_k):
        return []


def test_hybrid_retriever_uses_fuzzy_threshold():
    r = HybridRetriever(_Store(), entity_names=["alpha"])
    # High threshold rejects fuzzy-only matches; lower threshold accepts them.
    r.fuzzy_threshold = 200
    assert r.extract_entities("alp") == []
    r.fuzzy_threshold = 50
    assert r.extract_entities("alp") == ["alpha"]


def test_hybrid_retriever_default_threshold_from_config():
    r = HybridRetriever(_Store(), entity_names=["alpha"])
    assert r.fuzzy_threshold == Config().FUZZY_MATCH_THRESHOLD
