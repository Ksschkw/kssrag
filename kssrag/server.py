from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import time
import uuid
import json
from collections import OrderedDict
from functools import partial

import anyio

from .core.agents import RAGAgent
from .utils.helpers import logger
from .config import config

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    metadata_filter: Optional[Dict[str, Any]] = None

class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completions request."""
    model: Optional[str] = None
    messages: List[Dict[str, Any]]
    stream: bool = False
    user: Optional[str] = None

class ServerConfig(BaseModel):
    """Configuration for the FastAPI server"""
    host: str = config.SERVER_HOST
    port: int = config.SERVER_PORT
    cors_origins: List[str] = config.CORS_ORIGINS
    cors_allow_credentials: bool = config.CORS_ALLOW_CREDENTIALS
    cors_allow_methods: List[str] = config.CORS_ALLOW_METHODS
    cors_allow_headers: List[str] = config.CORS_ALLOW_HEADERS
    title: str = "KSSSwagger"
    description: str = "[kssrag](https://github.com/Ksschkw/kssrag)"
    version: str = "0.4.0"

def create_app(rag_agent: RAGAgent, server_config: Optional[ServerConfig] = None):
    """Create a FastAPI app for the RAG agent with configurable CORS"""
    if server_config is None:
        server_config = ServerConfig()
    
    app = FastAPI(
        title=server_config.title,
        description=server_config.description,
        version=server_config.version
    )
    
    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.cors_origins,
        allow_credentials=server_config.cors_allow_credentials,
        allow_methods=server_config.cors_allow_methods,
        allow_headers=server_config.cors_allow_headers,
    )
    
    # Session management (bounded LRU to prevent unbounded memory growth)
    sessions: "OrderedDict[str, RAGAgent]" = OrderedDict()
    max_sessions = config.MAX_SESSIONS

    def get_or_create_session(session_id: str) -> RAGAgent:
        """Return the agent for a session, creating it if needed, with LRU eviction."""
        if session_id in sessions:
            sessions.move_to_end(session_id)
            return sessions[session_id]

        logger.info(f"Creating new session: {session_id}")
        agent = RAGAgent(
            retriever=rag_agent.retriever,
            llm=rag_agent.llm,
            system_prompt=rag_agent.system_prompt,
        )
        sessions[session_id] = agent
        while len(sessions) > max_sessions:
            evicted_id, _ = sessions.popitem(last=False)
            logger.info(f"Evicted least-recently-used session: {evicted_id}")
        return agent

    @app.post("/query")
    async def query_endpoint(request: QueryRequest):
        """Handle user queries"""
        query = request.query
        session_id = request.session_id or str(uuid.uuid4())

        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        try:
            agent = get_or_create_session(session_id)
            # agent.query() is synchronous and network-bound; run it in a worker
            # thread so it does not block the event loop (allowing concurrency).
            response = await anyio.to_thread.run_sync(
                partial(
                    agent.query,
                    query,
                    top_k=config.TOP_K,
                    metadata_filter=request.metadata_filter,
                )
            )

            return {
                "query": query,
                "response": response,
                "session_id": session_id
            }

        except Exception:
            # Log the real failure server-side, but never leak internals to the client.
            logger.error("Error handling query", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.post("/stream")
    async def stream_query(request: QueryRequest):
        """Streaming query endpoint with Server-Sent Events"""
        query = request.query
        session_id = request.session_id or str(uuid.uuid4())

        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        try:
            agent = get_or_create_session(session_id)

            async def generate():
                # agent.query_stream is a *synchronous* generator that performs
                # blocking network I/O. Drain it in a worker thread and hand each
                # token back to the event loop via a queue, so streaming one client
                # never blocks the loop for the others.
                queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_running_loop()
                _DONE = object()

                def producer():
                    try:
                        for token in agent.query_stream(
                            query, top_k=config.TOP_K, metadata_filter=request.metadata_filter
                        ):
                            if token:
                                loop.call_soon_threadsafe(queue.put_nowait, ("chunk", token))
                        loop.call_soon_threadsafe(queue.put_nowait, ("done", None))
                    except Exception as exc:  # noqa: BLE001 - logged, generic error sent to client
                        logger.error(f"Streaming error: {exc}", exc_info=True)
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
                        else:  # error (payload is None; real error already logged in producer)
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
                }
            )

        except Exception as e:
            logger.error(f"Streaming query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatCompletionRequest):
        """OpenAI-compatible chat completions (plain and SSE streaming).

        Lets any OpenAI SDK/client talk to the kssrag server. The last user
        message is used as the retrieval query; conversation memory is keyed
        by the optional ``user`` field (defaults to a shared session).
        """
        if not request.messages:
            raise HTTPException(status_code=400, detail="messages is required")
        user_messages = [m for m in request.messages if m.get("role") == "user"]
        if not user_messages:
            raise HTTPException(status_code=400, detail="No user message found in messages")
        raw_content = user_messages[-1].get("content") or ""
        # Guard against multimodal (list) content; kssrag is text-only.
        query = raw_content if isinstance(raw_content, str) else str(raw_content)
        query = query.strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        session_id = request.user or "openai-default"
        try:
            agent = get_or_create_session(session_id)
            model = request.model or config.DEFAULT_MODEL

            if not request.stream:
                response = await anyio.to_thread.run_sync(
                    partial(agent.query, query, top_k=config.TOP_K)
                )
                return {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": model,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": response},
                        "finish_reason": "stop",
                    }],
                    # Rough character-based token estimates (no tokenizer here).
                    "usage": {
                        "prompt_tokens": len(query) // 4,
                        "completion_tokens": len(response) // 4,
                        "total_tokens": (len(query) + len(response)) // 4,
                    },
                }

            async def generate():
                queue: asyncio.Queue = asyncio.Queue()
                loop = asyncio.get_running_loop()
                completion_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
                created = int(time.time())

                def producer():
                    try:
                        for token in agent.query_stream(query, top_k=config.TOP_K):
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
                        "model": model,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                    }) + "\n\n"

                try:
                    yield delta_payload({"role": "assistant", "content": ""})
                    while True:
                        kind, payload = await queue.get()
                        if kind == "chunk":
                            yield delta_payload({"content": payload})
                        else:  # done or error: terminate the stream cleanly
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
        except Exception:
            logger.error("Error handling chat completion", exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy", 
            "message": "KSS RAG API is running",
            "version": server_config.version
        }
    
    @app.get("/config")
    async def get_config():
        """Get current server configuration"""
        return server_config.model_dump()
    
    def _clear_session(session_id: str):
        """Clear a session's conversation history (shared by POST and legacy GET)."""
        if session_id in sessions:
            sessions[session_id].clear_conversation()
            return {"message": f"Session {session_id} cleared"}
        raise HTTPException(status_code=404, detail="Session not found")

    @app.post("/sessions/{session_id}/clear")
    async def clear_session(session_id: str):
        """Clear a session's conversation history."""
        return _clear_session(session_id)

    @app.delete("/sessions/{session_id}")
    async def delete_session(session_id: str):
        """Delete a session entirely (drops its agent and conversation)."""
        if sessions.pop(session_id, None) is not None:
            return {"message": f"Session {session_id} deleted"}
        raise HTTPException(status_code=404, detail="Session not found")

    @app.get("/sessions/{session_id}/clear", deprecated=True)
    async def clear_session_legacy(session_id: str):
        """Deprecated: use POST /sessions/{id}/clear. Kept for backward compatibility."""
        logger.warning(
            "GET /sessions/{id}/clear is deprecated and mutates state; "
            "use POST /sessions/{id}/clear instead."
        )
        return _clear_session(session_id)
    
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Welcome to KSSRAG API",
            "version": server_config.version,
            "docs": "/docs",
            "health": "/health"
        }
    
    return app, server_config