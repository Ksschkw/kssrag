"""
Correlation ID middleware.
Conforms to house-standard.md §9.
"""
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..core.errors import set_correlation_id


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Generates or propagates X-Correlation-ID and binds it to contextvars."""

    async def dispatch(self, request: Request, call_next) -> Response:
        # Extract existing correlation ID header or generate a new one
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = str(uuid.uuid4())

        set_correlation_id(correlation_id)
        request.state.correlation_id = correlation_id

        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        return response
