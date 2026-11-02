"""
Global error handler middleware.
Conforms to house-standard.md §9 and §10.
"""
import traceback
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..core.errors import KSSRAGError, get_correlation_id
from ..core.logging import logger


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Catches domain and unexpected exceptions, logs full internal view, and returns safe external view."""

    async def dispatch(self, request: Request, call_next) -> Response:
        try:
            return await call_next(request)
        except KSSRAGError as domain_error:
            # Internal logging with full diagnostics (§9)
            cid = domain_error.correlation_id or get_correlation_id()
            logger.error(
                f"[ERROR] operation={domain_error.operation} entity={domain_error.entity} "
                f"identifier={domain_error.identifier} layer={domain_error.layer} "
                f"cause={domain_error.cause} message={domain_error.message}"
            )
            external_body = domain_error.to_external_response()
            # Map domain error to status code
            status_code = 400
            if domain_error.__class__.__name__ == "NotFoundError":
                status_code = 404
            elif domain_error.__class__.__name__ == "ConflictError":
                status_code = 409
            elif domain_error.__class__.__name__ == "ProviderUnavailableError":
                status_code = 503
            elif domain_error.__class__.__name__ == "RateLimitExceededError":
                status_code = 429
            elif domain_error.__class__.__name__ == "SecurityViolationError":
                status_code = 403

            return JSONResponse(
                status_code=status_code,
                content=external_body,
                headers={"X-Correlation-ID": cid},
            )
        except Exception as unhandled:
            cid = get_correlation_id()
            # Log full stack trace internally for engineers
            logger.error(
                f"[UNHANDLED_EXCEPTION] correlation_id={cid} error={unhandled} "
                f"trace={traceback.format_exc()}"
            )
            # Never leak internal stack trace, file path, or exception name to client (§9, §10)
            return JSONResponse(
                status_code=500,
                content={
                    "error": {
                        "message": "We could not complete your request due to an internal issue.",
                        "action": "Please wait a moment and retry your request. If this continues, contact support quoting this reference.",
                        "reference": cid,
                    }
                },
                headers={"X-Correlation-ID": cid},
            )
