"""
Security headers middleware.
Conforms to house-standard.md §11.
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from ..core.security import DEFAULT_SECURITY_HEADERS


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Applies standard defense-in-depth HTTP security headers to all responses."""

    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        for header, value in DEFAULT_SECURITY_HEADERS.items():
            # Set header if not already customized by the endpoint
            if header not in response.headers:
                response.headers[header] = value
        return response
