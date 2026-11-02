"""
Query router.
Conforms to house-standard.md §2 (Parse request, call one service, shape response).
"""
import asyncio
import json
import uuid
from functools import partial
import anyio
from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import StreamingResponse

from ..schemas.query_schema import QueryRequestSchema, QueryResponseSchema
from ..core.config import config
from ..core.logging import logger

query_router = APIRouter()


@query_router.post("/query", response_model=QueryResponseSchema)
async def query_endpoint(request_body: QueryRequestSchema, request: Request):
    """Handle user query."""
    rag_service = request.app.state.rag_service
    session_id = request_body.session_id or str(uuid.uuid4())

    # Offload blocking synchronous query to worker thread
    result = await anyio.to_thread.run_sync(
        partial(
            rag_service.execute_query,
            query=request_body.query,
            session_id=session_id,
            top_k=config.TOP_K,
            metadata_filter=request_body.metadata_filter,
        )
    )

    return QueryResponseSchema(
        query=result.query,
        response=result.response,
        session_id=result.session_id,
    )


@query_router.post("/stream")
async def stream_query_endpoint(request_body: QueryRequestSchema, request: Request):
    """Streaming query endpoint with Server-Sent Events."""
    rag_service = request.app.state.rag_service

    async def generate():
        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def producer():
            try:
                for token in rag_service.execute_stream(
                    query=request_body.query,
                    top_k=config.TOP_K,
                    metadata_filter=request_body.metadata_filter,
                ):
                    if token:
                        loop.call_soon_threadsafe(queue.put_nowait, ("chunk", token))
                loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
            except Exception as exc:
                logger.error(f"Streaming error in producer: {exc}", exc_info=True)
                loop.call_soon_threadsafe(queue.put_nowait, ("error", None))

        producer_task = anyio.to_thread.run_sync(producer)
        producer_future = asyncio.ensure_future(producer_task)
        try:
            while True:
                kind, payload = await queue.get()
                if kind == "chunk":
                    yield f"data: {json.dumps({'chunk': payload, 'done': False})}\n\n"
                elif kind == "done":
                    yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
                    break
                else:
                    yield f"data: {json.dumps({'error': 'stream failed', 'done': True})}\n\n"
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
