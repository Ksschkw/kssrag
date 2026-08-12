"""
Server endpoint tests (no real LLM, no network).

Exercises the FastAPI app end-to-end with a fake agent injected, covering the
query/stream endpoints, health/config, session lifecycle, LRU eviction, and
input validation. Uses Starlette's TestClient (httpx-backed).
"""
import json

import pytest

pytest.importorskip("httpx")  # TestClient needs httpx

from starlette.testclient import TestClient

from kssrag.core.agents import RAGAgent
from kssrag.server import create_app, ServerConfig
from kssrag.config import config as global_config


class _FakeRetriever:
    def retrieve(self, query, top_k=5, metadata_filter=None):
        return [{"content": "ctx", "metadata": {"id": 0}}]


class _FakeLLM:
    def predict(self, messages):
        return "fake answer"

    def predict_stream(self, messages):
        for tok in ["fake", " ", "answer"]:
            yield tok


def _make_client(max_sessions=None):
    if max_sessions is not None:
        global_config.MAX_SESSIONS = max_sessions
    agent = RAGAgent(_FakeRetriever(), _FakeLLM(), system_prompt="sys")
    app, _ = create_app(agent)
    return TestClient(app)


def test_health():
    client = _make_client()
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_config_endpoint():
    client = _make_client()
    r = client.get("/config")
    assert r.status_code == 200
    assert "version" in r.json()


def test_root():
    client = _make_client()
    assert client.get("/").status_code == 200


def test_query_returns_answer():
    client = _make_client()
    r = client.post("/query", json={"query": "hello", "session_id": "s1"})
    assert r.status_code == 200
    body = r.json()
    assert body["response"] == "fake answer"
    assert body["session_id"] == "s1"


def test_query_empty_rejected():
    client = _make_client()
    r = client.post("/query", json={"query": "   "})
    assert r.status_code == 400


def test_query_autogenerates_session_id():
    client = _make_client()
    r = client.post("/query", json={"query": "hello"})
    assert r.status_code == 200
    assert r.json()["session_id"]  # a uuid was assigned


def test_stream_sse():
    client = _make_client()
    with client.stream("POST", "/stream", json={"query": "hi", "session_id": "s"}) as r:
        assert r.status_code == 200
        chunks = []
        done = False
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = json.loads(line[6:])
            if payload.get("done"):
                done = True
            elif payload.get("chunk"):
                chunks.append(payload["chunk"])
    assert "".join(chunks) == "fake answer"
    assert done


def test_session_clear_post():
    client = _make_client()
    client.post("/query", json={"query": "hi", "session_id": "sc"})
    r = client.post("/sessions/sc/clear")
    assert r.status_code == 200


def test_session_clear_missing_404():
    client = _make_client()
    assert client.post("/sessions/nope/clear").status_code == 404


def test_session_delete():
    client = _make_client()
    client.post("/query", json={"query": "hi", "session_id": "sd"})
    assert client.delete("/sessions/sd").status_code == 200
    assert client.delete("/sessions/sd").status_code == 404  # already gone


def test_legacy_get_clear_still_works():
    client = _make_client()
    client.post("/query", json={"query": "hi", "session_id": "sl"})
    assert client.get("/sessions/sl/clear").status_code == 200


def test_session_lru_eviction():
    client = _make_client(max_sessions=3)
    for i in range(5):
        client.post("/query", json={"query": "hi", "session_id": f"e{i}"})
    # With cap 3, only the 3 most-recent sessions should remain.
    present = [i for i in range(5) if client.post(f"/sessions/e{i}/clear").status_code == 200]
    assert present == [2, 3, 4]


# --- TOP_K + metadata_filter pass-through ------------------------------------

class _RecordingRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k=5, metadata_filter=None):
        self.calls.append((query, top_k, metadata_filter))
        return [{"content": "ctx", "metadata": {"id": 0}}]


def test_query_honors_config_top_k_and_filter(monkeypatch):
    monkeypatch.setattr(global_config, "TOP_K", 7)
    rr = _RecordingRetriever()
    agent = RAGAgent(rr, _FakeLLM(), system_prompt="sys")
    app, _ = create_app(agent)
    client = TestClient(app)

    r = client.post("/query", json={"query": "hello", "metadata_filter": {"source": "x"}})
    assert r.status_code == 200
    query, top_k, metadata_filter = rr.calls[0]
    assert top_k == 7
    assert metadata_filter == {"source": "x"}


def test_stream_honors_config_top_k(monkeypatch):
    monkeypatch.setattr(global_config, "TOP_K", 4)
    rr = _RecordingRetriever()
    agent = RAGAgent(rr, _FakeLLM(), system_prompt="sys")
    app, _ = create_app(agent)
    client = TestClient(app)

    with client.stream("POST", "/stream", json={"query": "hi"}) as r:
        assert r.status_code == 200
        for _ in r.iter_lines():
            pass
    assert rr.calls[0][1] == 4  # top_k


# --- OpenAI-compatible /v1/chat/completions ---------------------------------

def test_chat_completions_plain():
    client = _make_client()
    r = client.post("/v1/chat/completions", json={
        "model": "test-model",
        "messages": [{"role": "system", "content": "sys"},
                      {"role": "user", "content": "hello"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["model"] == "test-model"
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert body["choices"][0]["message"]["content"] == "fake answer"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "usage" in body


def test_chat_completions_stream_openai_events():
    client = _make_client()
    with client.stream("POST", "/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    }) as r:
        assert r.status_code == 200
        events = [l[6:] for l in r.iter_lines() if l.startswith("data: ")]

    assert events[-1] == "[DONE]"
    parsed = [json.loads(e) for e in events[:-1]]
    assert parsed[0]["object"] == "chat.completion.chunk"
    # When the client omits `model`, the configured DEFAULT_MODEL is echoed.
    assert parsed[0]["model"] == global_config.DEFAULT_MODEL
    assert parsed[0]["choices"][0]["delta"]["role"] == "assistant"
    content = "".join(
        p["choices"][0]["delta"].get("content", "") for p in parsed
    )
    assert content == "fake answer"
    assert parsed[-1]["choices"][0]["finish_reason"] == "stop"


def test_chat_completions_requires_user_message():
    client = _make_client()
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "system", "content": "x"}]
    })
    assert r.status_code == 400


def test_chat_completions_empty_messages():
    client = _make_client()
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400
