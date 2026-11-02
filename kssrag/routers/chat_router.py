"""
OpenAI-compatible chat completion router.
Conforms to house-standard.md §2 (Parse request, call one service, shape response).
"""
import asyncio
import json
import time
import uuid
from functools import partial
import anyio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from ..schemas.chat_schema import (
    ChatCompletionRequestSchema,
    ChatCompletionResponseSchema,
    ChatChoiceSchema,
    ChatMessageSchema,
    ChatUsageSchema,
)
from ..core.config import config
from ..core.logging import logger

chat_router = APIRouter()


@chat_router.post("/v1/chat/completions")
async def chat_completions_endpoint(request_body: ChatCompletionRequestSchema, request: Request):
    """Handle OpenAI-compatible chat completions."""
    rag_service = request.app.state.rag_service
    user_messages = [m for m in request_body.messages if m.get("role") == "user"]
    raw_content = user_messages[-1].get("content", "")
    query = (raw_content if isinstance(raw_content, str) else str(raw_content)).strip()
    session_id = request_body.user or "openai-default"
    model_name = request_body.model or config.DEFAULT_MODEL

    if not request_body.stream:
        result = await anyio.to_thread.run_sync(
            partial(
                rag_service.execute_query,
                query=query,
                session_id=session_id,
                top_k=config.TOP_K,
            )
        )

        response_text = result.response
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": len(query) // 4,
                "completion_tokens": len(response_text) // 4,
                "total_tokens": (len(query) + len(response_text)) // 4,
            },
        }

    async def generate():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()
        completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
        created = int(time.time())

        def producer():
            try:
                for token in rag_service.execute_stream(query=query, top_k=config.TOP_K):
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, ("chunk", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception:
                logger.error("OpenAI streaming failed", exc_info=True)
                loop.call_soon_threadsafe(queue.put_nowait, ("error", None))

        producer_future = asyncio.ensure_future(anyio.to_thread.run_sync(producer))

        def delta_payload(delta: dict, finish_reason=None) -> str:
            return "data: " + json.dumps({
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }) + "\n\n"

        try:
            yield delta_payload({"role": "assistant", "content": ""})
            while True:
                kind, payload = await queue.get()
                if kind == "chunk":
                    yield delta_payload({"content": payload})
                else:
                    yield delta_payload({}, finish_reason="stop")
                    yield "data: [DONE]\n\n"
                    break
        finally:
            await producer_future

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
