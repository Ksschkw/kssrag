"""
OpenAI SDK compatibility tests for /v1/chat/completions.

Drives the endpoint with the REAL `openai` Python SDK — the strictest possible
client, since it validates the full response schema. No network and no live
server: the SDK's httpx client is pointed at the in-process ASGI app via
httpx's ASGITransport. Skipped entirely if `openai` isn't installed (it's a
dev-only dependency, see setup.py's dev extra).
"""
import asyncio

import pytest

pytest.importorskip("openai")

import httpx
from openai import AsyncOpenAI

from kssrag.core.agents import RAGAgent
from kssrag.server import create_app


class _FakeRetriever:
    def retrieve(self, query, top_k=5, metadata_filter=None):
        return [{"content": "ctx", "metadata": {"id": 0}}]


class _FakeLLM:
    def __init__(self):
        self.calls = []

    def predict(self, messages):
        self.calls.append(messages)
        return "fake answer"

    def predict_stream(self, messages):
        self.calls.append(messages)
        for tok in ["fake", " ", "answer"]:
            yield tok


def _make_sdk_client(app):
    """Return an AsyncOpenAI client routed to the in-process ASGI app."""
    transport = httpx.ASGITransport(app=app)
    http_client = httpx.AsyncClient(transport=transport)
    return AsyncOpenAI(
        api_key="sk-test-not-real",
        base_url="http://test/v1",  # host is ignored by ASGITransport
        http_client=http_client,
    )


def test_sdk_plain_completion_parses():
    agent = RAGAgent(_FakeRetriever(), _FakeLLM(), system_prompt="sys")
    app, _ = create_app(agent)

    async def _run():
        client = _make_sdk_client(app)
        try:
            return await client.chat.completions.create(
                model="test-model",
                messages=[
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "hello"},
                ],
            )
        finally:
            await client.close()

    resp = asyncio.run(_run())
    assert resp.object == "chat.completion"
    assert resp.model == "test-model"
    assert resp.choices[0].message.role == "assistant"
    assert resp.choices[0].message.content == "fake answer"
    assert resp.choices[0].finish_reason == "stop"
    assert resp.usage is not None
    assert resp.id.startswith("chatcmpl-")


def test_sdk_streaming_reassembles():
    llm = _FakeLLM()
    agent = RAGAgent(_FakeRetriever(), llm, system_prompt="sys")
    app, _ = create_app(agent)

    async def _run():
        client = _make_sdk_client(app)
        try:
            stream = await client.chat.completions.create(
                model="test-model",
                messages=[{"role": "user", "content": "hi"}],
                stream=True,
            )
            content = []
            last_reason = None
            models = set()
            async for chunk in stream:
                models.add(chunk.model)
                if chunk.choices[0].delta.content:
                    content.append(chunk.choices[0].delta.content)
                if chunk.choices[0].finish_reason:
                    last_reason = chunk.choices[0].finish_reason
            return "".join(content), last_reason, models
        finally:
            await client.close()

    content, last_reason, models = asyncio.run(_run())
    assert content == "fake answer"
    assert last_reason == "stop"
    assert "test-model" in models


def test_sdk_sessions_carry_conversation():
    """Requests with the same `user` share conversation memory; different ones don't."""
    llm = _FakeLLM()
    agent = RAGAgent(_FakeRetriever(), llm, system_prompt="sys")
    app, _ = create_app(agent)

    async def _run():
        client = _make_sdk_client(app)
        try:
            await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "one"}], user="alice"
            )
            await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "two"}], user="alice"
            )
            await client.chat.completions.create(
                model="m", messages=[{"role": "user", "content": "three"}], user="bob"
            )
        finally:
            await client.close()

    asyncio.run(_run())
    assert len(llm.calls) == 3
    # RAGAgent stores assistant replies in conversation; each fresh question is
    # appended at call time. So alice's 2nd call sees her 1st assistant reply in
    # history, while bob's fresh session sees none.
    assert len([m for m in llm.calls[0] if m["role"] == "assistant"]) == 0
    assert len([m for m in llm.calls[1] if m["role"] == "assistant"]) == 1
    assert len([m for m in llm.calls[2] if m["role"] == "assistant"]) == 0
    # ...and each call still carries exactly one user question (its own).
    assert all(len([m for m in call if m["role"] == "user"]) == 1 for call in llm.calls)
