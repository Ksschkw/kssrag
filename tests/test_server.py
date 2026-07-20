"""
Server endpoint tests (no real LLM, no network).

Exercises the FastAPI app end-to-end with a fake agent injected, covering the
query/stream endpoints, health/config, session lifecycle, LRU eviction, and
input validation. Uses Starlette's TestClient (httpx-backed).
"""
import json

import pytest

pytest.importorskip("httpx")  # TestClient needs httpx

from kssrag.core.agents import RAGAgent
from kssrag.server import create_app, ServerConfig
from kssrag.config import config as global_config


class _FakeRetriever:
    def retrieve(self, query, top_k=5):
        return [{"content": "ctx", "metadata": {"id": 0}}]


class _FakeLLM:
    def predict(self, messages):
        return "fake answer"

    def predict_stream(self, messages):
        for tok in ["fake", " ", "answer"]:
            yield tok


def _make_client(max_sessions=None):
    from starlette.testclient import TestClient
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
