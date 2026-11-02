"""Rate limiter behaviour.

These tests exist because the limiter had no coverage at all, its per-client
map grew without bound, and it counted every request behind a load balancer
against the balancer's address.
"""
import asyncio

from starlette.requests import Request
from starlette.responses import Response

from kssrag.middleware.rate_limit_middleware import RateLimitMiddleware


def _request(path="/query", peer="203.0.113.7", forwarded=None):
    """Build a request scope with a chosen peer address."""
    headers = []
    if forwarded is not None:
        headers.append((b"x-forwarded-for", forwarded.encode("ascii")))
    scope = {
        "type": "http",
        "method": "POST",
        "path": path,
        "query_string": b"",
        "headers": headers,
        "client": (peer, 54321),
        "server": ("testserver", 80),
        "scheme": "http",
    }
    return Request(scope)


async def _ok(request):
    """Stand-in for the downstream application."""
    return Response("ok", status_code=200)


def _call(middleware, request):
    """Dispatch one request through the middleware."""
    return asyncio.run(middleware.dispatch(request, _ok))


def _middleware(**kwargs):
    """Build a limiter with a downstream app that always succeeds."""
    return RateLimitMiddleware(app=lambda scope, receive, send: None, **kwargs)


def test_requests_under_the_limit_pass():
    """The first requests from a client are served."""
    middleware = _middleware(requests_per_minute=3)

    responses = [_call(middleware, _request()) for _ in range(3)]

    assert [response.status_code for response in responses] == [200, 200, 200]


def test_request_over_the_limit_is_rejected_with_retry_after():
    """One request past the limit returns 429 with an actionable body."""
    middleware = _middleware(requests_per_minute=2)
    _call(middleware, _request())
    _call(middleware, _request())

    response = _call(middleware, _request())

    assert response.status_code == 429
    assert response.headers["Retry-After"].isdigit()
    assert int(response.headers["Retry-After"]) >= 1


def test_tracked_clients_never_exceed_the_cap():
    """Memory stays bounded no matter how many distinct clients appear."""
    middleware = _middleware(requests_per_minute=5, max_tracked_clients=10)

    for index in range(200):
        _call(middleware, _request(peer=f"198.51.100.{index}"))

    assert middleware.tracked_clients() <= 10


def test_forwarded_header_is_ignored_when_not_trusted():
    """A caller cannot escape the limit by inventing an X-Forwarded-For value."""
    middleware = _middleware(requests_per_minute=2, trust_forwarded_for=False)

    _call(middleware, _request(forwarded="1.1.1.1"))
    _call(middleware, _request(forwarded="2.2.2.2"))
    response = _call(middleware, _request(forwarded="3.3.3.3"))

    assert response.status_code == 429
    assert middleware.tracked_clients() == 1


def test_forwarded_header_separates_clients_when_trusted():
    """Behind a trusted proxy each forwarded address keeps its own bucket."""
    middleware = _middleware(requests_per_minute=2, trust_forwarded_for=True)

    for forwarded in ("1.1.1.1", "2.2.2.2", "3.3.3.3"):
        response = _call(middleware, _request(forwarded=forwarded))
        assert response.status_code == 200

    assert middleware.tracked_clients() == 3


def test_health_is_exempt_from_limiting():
    """Liveness probes are never rate limited."""
    middleware = _middleware(requests_per_minute=1)

    responses = [_call(middleware, _request(path="/health")) for _ in range(5)]

    assert [response.status_code for response in responses] == [200] * 5
    assert middleware.tracked_clients() == 0
