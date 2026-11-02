"""
Rate limiting middleware.

Conforms to house-standard.md section 11 (resource limits).

The limiter keeps a sliding one-minute window per client identity. Two
properties matter for a public deployment:

1. Memory is bounded. The previous version used a defaultdict keyed by client
   IP and never removed a key, so every distinct address that ever made one
   request stayed in memory for the life of the process. The number of tracked
   identities is now capped and the least recently seen one is evicted.

2. The client identity is the socket peer unless the service is told it sits
   behind a trusted proxy. Behind a platform load balancer the peer address is
   the balancer, so every user would otherwise share a single bucket.
"""
import time
from collections import OrderedDict
from typing import Dict, List

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from ..core.errors import get_correlation_id

WINDOW_SECONDS = 60.0

EXEMPT_PATHS = frozenset({"/", "/health", "/docs", "/openapi.json", "/redoc"})


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding-window rate limiter with bounded memory and optional proxy trust."""

    def __init__(
        self,
        app,
        requests_per_minute: int = 120,
        max_tracked_clients: int = 10000,
        trust_forwarded_for: bool = False,
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.max_tracked_clients = max_tracked_clients
        self.trust_forwarded_for = trust_forwarded_for
        self._windows: "OrderedDict[str, List[float]]" = OrderedDict()

    def client_key(self, request: Request) -> str:
        """Identity the request is counted against.

        With trust_forwarded_for disabled, a client-supplied X-Forwarded-For is
        ignored: honouring it by default would let any caller evade the limit by
        inventing a new address per request.
        """
        if self.trust_forwarded_for:
            forwarded = request.headers.get("x-forwarded-for", "")
            first_entry = forwarded.split(",")[0].strip()
            if first_entry:
                return first_entry
        if request.client is not None:
            return request.client.host
        return "unknown-client"

    def _recent_requests(self, key: str, now: float) -> List[float]:
        """Timestamps for ``key`` still inside the window."""
        window_start = now - WINDOW_SECONDS
        return [stamp for stamp in self._windows.get(key, []) if stamp > window_start]

    def _remember(self, key: str, stamps: List[float]) -> None:
        """Store the window for ``key``, evicting the least recently seen key."""
        self._windows[key] = stamps
        self._windows.move_to_end(key)
        while len(self._windows) > self.max_tracked_clients:
            self._windows.popitem(last=False)

    def tracked_clients(self) -> int:
        """Number of client identities currently held. Exposed for tests."""
        return len(self._windows)

    async def dispatch(self, request: Request, call_next) -> Response:
        """Apply the limit, then delegate."""
        if request.url.path in EXEMPT_PATHS:
            return await call_next(request)

        now = time.time()
        key = self.client_key(request)
        stamps = self._recent_requests(key, now)

        if len(stamps) >= self.requests_per_minute:
            self._remember(key, stamps)
            correlation_id = get_correlation_id()
            retry_after = max(1, int(round(WINDOW_SECONDS - (now - stamps[0]))))
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "message": "You have submitted too many requests in a short period.",
                        "action": "Please pause for a few seconds before submitting another request.",
                        "reference": correlation_id,
                    }
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-Correlation-ID": correlation_id,
                },
            )

        stamps.append(now)
        self._remember(key, stamps)
        return await call_next(request)
