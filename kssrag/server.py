"""
FastAPI Server for KSS RAG.

This module is the composition root: it builds the app, wires middleware, and
injects the services the routers call. It holds no business logic.
"""
from typing import Optional, List

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .core.agents import RAGAgent
from .core.config import config
from .core.container import Container
from .core.logging import logger
from .core.security import validate_cors_configuration
from .crud.session_crud import SessionCrud
from .services.rag_service import RAGService
from .services.session_service import SessionService
from .schemas.query_schema import QueryRequestSchema as QueryRequest
from .schemas.chat_schema import ChatCompletionRequestSchema as ChatCompletionRequest
from .middleware.correlation_middleware import CorrelationMiddleware
from .middleware.security_headers_middleware import SecurityHeadersMiddleware
from .middleware.rate_limit_middleware import RateLimitMiddleware
from .middleware.error_handler_middleware import ErrorHandlerMiddleware
from .routers.query_router import query_router
from .routers.chat_router import chat_router
from .routers.session_router import session_router
from .routers.health_router import health_router


class ServerConfig(BaseModel):
    """Configuration for the FastAPI server."""
    host: str = config.SERVER_HOST
    port: int = config.SERVER_PORT
    cors_origins: List[str] = config.CORS_ORIGINS
    cors_allow_credentials: bool = config.CORS_ALLOW_CREDENTIALS
    cors_allow_methods: List[str] = config.CORS_ALLOW_METHODS
    cors_allow_headers: List[str] = config.CORS_ALLOW_HEADERS
    title: str = "KSSSwagger"
    description: str = "[kssrag](https://github.com/Ksschkw/kssrag)"
    version: str = "0.4.1"


def create_app(rag_agent: RAGAgent, server_config: Optional[ServerConfig] = None):
    """Build the FastAPI app for ``rag_agent``.

    Returns ``(app, server_config)``. Every collaborator is constructed here and
    injected; routers read them from ``app.state``.
    """
    if server_config is None:
        server_config = ServerConfig()

    app = FastAPI(
        title=server_config.title,
        description=server_config.description,
        version=server_config.version,
    )

    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
    from .core.errors import get_correlation_id

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request, exc):
        cid = get_correlation_id()
        # Actionable message without leaking internals (section 10).
        errors = exc.errors()
        field = errors[0].get("loc", ["input"])[-1] if errors else "input"
        msg = errors[0].get("msg", "invalid value") if errors else "invalid value"
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": f"The value for '{field}' is invalid: {msg}",
                    "action": f"Provide a valid {field} and submit again.",
                    "reference": cid,
                },
                "detail": str(exc),
            },
            headers={"X-Correlation-ID": cid},
        )

    # Validate the CORS allowlist (section 11).
    try:
        validate_cors_configuration(
            origins=server_config.cors_origins,
            allow_credentials=server_config.cors_allow_credentials,
        )
    except Exception as e:
        logger.warning(f"CORS validation warning: {e}")

    # Middleware. Starlette runs the last added outermost.
    # 1. Error handler: converts domain errors to safe responses.
    app.add_middleware(ErrorHandlerMiddleware)
    # 2. Rate limiting: per-client protection with bounded memory.
    app.add_middleware(
        RateLimitMiddleware,
        requests_per_minute=config.RATE_LIMIT_PER_MINUTE,
        max_tracked_clients=config.RATE_LIMIT_MAX_TRACKED_CLIENTS,
        trust_forwarded_for=config.TRUST_FORWARDED_FOR,
    )
    # 3. Security headers: nosniff, frame options, HSTS.
    app.add_middleware(SecurityHeadersMiddleware)
    # 4. Correlation ID: threads X-Correlation-ID through the request.
    app.add_middleware(CorrelationMiddleware)
    # 5. CORS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.cors_origins,
        allow_credentials=server_config.cors_allow_credentials,
        allow_methods=server_config.cors_allow_methods,
        allow_headers=server_config.cors_allow_headers,
    )

    # Composition: every service is constructed here and injected.
    def create_session_agent() -> RAGAgent:
        return RAGAgent(
            retriever=rag_agent.retriever,
            llm=rag_agent.llm,
            system_prompt=rag_agent.system_prompt,
        )

    settings = Container()
    session_crud = SessionCrud(
        agent_factory=create_session_agent,
        max_sessions=config.MAX_SESSIONS,
    )
    rag_service = RAGService(
        session_crud=session_crud,
        circuit_breaker=settings.get_circuit_breaker("outbound_llm"),
    )
    session_service = SessionService(session_crud)

    app.state.server_config = server_config
    app.state.session_crud = session_crud
    app.state.rag_service = rag_service
    app.state.session_service = session_service

    app.include_router(query_router)
    app.include_router(chat_router)
    app.include_router(session_router)
    app.include_router(health_router)

    return app, server_config
