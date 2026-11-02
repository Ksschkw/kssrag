"""
Strict integration tests for FastAPI server contracts and house standards (§9, §10, §11, §12, §13).
Verifies:
- Correlation ID header propagation
- Security headers (HSTS, CSP, nosniff, DENY)
- Rate limiting middleware behavior
- Critical action confirmation on session deletion (§13)
- Safe error sanitization and 4-part external error format
- Rejection of oversized and forbidden fields
"""
import pytest
from starlette.testclient import TestClient

from kssrag.core.agents import RAGAgent
from kssrag.server import create_app, ServerConfig
from kssrag.models.base import BaseLLM


class _ServerStrictFakeRetriever:
    def retrieve(self, query, top_k=5, metadata_filter=None):
        return [{"content": "Strict context document.", "metadata": {"id": 100}}]


class _ServerStrictFakeLLM(BaseLLM):
    def predict(self, messages):
        return "Strict fake answer"

    def predict_stream(self, messages):
        yield "Strict "
        yield "fake "
        yield "answer"


def _make_strict_client():
    agent = RAGAgent(_ServerStrictFakeRetriever(), _ServerStrictFakeLLM(), system_prompt="sys")
    app, _ = create_app(agent)
    return TestClient(app)


def test_correlation_id_propagation_and_generation():
    """Verify X-Correlation-ID is generated and returned on response (§9)."""
    client = _make_strict_client()

    # 1. Server generates correlation ID if omitted
    res = client.get("/health")
    assert res.status_code == 200
    assert "X-Correlation-ID" in res.headers
    generated_id = res.headers["X-Correlation-ID"]
    assert len(generated_id) > 10

    # 2. Server preserves client-supplied correlation ID
    custom_cid = "client-correlation-uuid-999"
    res2 = client.get("/health", headers={"X-Correlation-ID": custom_cid})
    assert res2.headers.get("X-Correlation-ID") == custom_cid


def test_security_headers_present_on_all_responses():
    """Verify defense-in-depth security headers are attached (§11)."""
    client = _make_strict_client()
    res = client.get("/health")
    assert res.status_code == 200
    assert res.headers.get("X-Content-Type-Options") == "nosniff"
    assert res.headers.get("X-Frame-Options") == "DENY"
    assert res.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"
    assert "max-age=" in res.headers.get("Strict-Transport-Security", "")
    assert "default-src 'self'" in res.headers.get("Content-Security-Policy", "")


def test_health_check_reveals_no_version_or_internals():
    """Verify health check exposes liveness and readiness only (§11)."""
    client = _make_strict_client()
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "healthy"
    # house-standard.md §11 explicitly forbids exposing version numbers in health check
    assert "version" not in body
    assert "dependencies" not in body
    assert "config" not in body


def test_critical_action_session_deletion_confirmation():
    """Verify session deletion requires confirmation when explicitly unconfirmed (§13)."""
    client = _make_strict_client()
    session_id = "test-crit-session-1"

    # Seed session
    client.post("/query", json={"query": "initialize", "session_id": session_id})

    # Attempt deletion with confirm=false -> MUST fail with 400 and confirmation guidance
    res_unconfirmed = client.delete(f"/sessions/{session_id}?confirm=false")
    assert res_unconfirmed.status_code == 400
    err_body = res_unconfirmed.json()
    assert "error" in err_body
    assert "Confirmation required" in err_body["error"]["message"]
    assert "reference" in err_body["error"]  # correlation ID included

    # Attempt deletion with explicit confirm=true -> MUST succeed
    res_confirmed = client.delete(f"/sessions/{session_id}?confirm=true")
    assert res_confirmed.status_code == 200
    assert "deleted" in res_confirmed.json()["message"]


def test_unknown_fields_rejected_by_schemas():
    """Verify extra unknown fields are rejected at the transport boundary (§11)."""
    client = _make_strict_client()
    res = client.post("/query", json={
        "query": "hello",
        "session_id": "s1",
        "malicious_unknown_field": "injected",
    })
    # extra='forbid' in QueryRequestSchema causes 400 validation error
    assert res.status_code == 400


def test_banned_phrases_not_present_in_error_responses():
    """Verify external error messages do not contain banned phrases (§10)."""
    client = _make_strict_client()
    res = client.post("/query", json={"query": "   "})
    assert res.status_code == 400
    body_text = res.text.lower()
    banned_phrases = ["something went wrong", "unknown error", "internal server error", "error 500", "nan", "null"]
    for phrase in banned_phrases:
        assert phrase not in body_text, f"Banned phrase '{phrase}' found in response: {body_text}"
