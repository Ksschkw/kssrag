"""
Outbound resilience primitives for KSS RAG: Circuit Breaker, Bulkhead, Jittered Retries.
Conforms to house-standard.md §7.
"""
import random
import threading
import time
from typing import Callable, Any, Optional, TypeVar, Tuple

from .errors import ProviderUnavailableError
from .logging import logger

T = TypeVar("T")


class CircuitBreakerOpenError(ProviderUnavailableError):
    """Raised immediately when a call is made through an OPEN circuit breaker."""


class CircuitBreaker:
    """Thread-safe circuit breaker with CLOSED, OPEN, and HALF_OPEN states (§7).

    - CLOSED: Normal operation. Every failure increments failure counter.
    - OPEN: Tripped after `failure_threshold` consecutive failures. Outbound calls fail fast.
    - HALF_OPEN: Entered after `recovery_timeout_seconds`. Allows `probe_count` test calls.
    """

    STATE_CLOSED = "CLOSED"
    STATE_OPEN = "OPEN"
    STATE_HALF_OPEN = "HALF_OPEN"

    def __init__(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
        probe_count: int = 2,
    ):
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.probe_count = probe_count

        self._state = self.STATE_CLOSED
        self._failure_count = 0
        self._successful_probes = 0
        self._last_state_change = time.time()
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            # Check for transition from OPEN to HALF_OPEN
            if self._state == self.STATE_OPEN:
                if (time.time() - self._last_state_change) >= self.recovery_timeout_seconds:
                    self._transition_to(self.STATE_HALF_OPEN)
            return self._state

    def _transition_to(self, new_state: str) -> None:
        old_state = self._state
        self._state = new_state
        self._last_state_change = time.time()
        logger.info(
            f"[RESILIENCE] CircuitBreaker '{self.name}' state transition: "
            f"{old_state} -> {new_state} (failures: {self._failure_count})"
        )

    def record_success(self) -> None:
        """Record successful outbound response."""
        with self._lock:
            if self._state == self.STATE_HALF_OPEN:
                self._successful_probes += 1
                if self._successful_probes >= self.probe_count:
                    self._failure_count = 0
                    self._successful_probes = 0
                    self._transition_to(self.STATE_CLOSED)
            elif self._state == self.STATE_CLOSED:
                self._failure_count = 0

    def record_failure(self, error: Optional[Exception] = None) -> None:
        """Record outbound failure."""
        with self._lock:
            self._failure_count += 1
            if self._state == self.STATE_HALF_OPEN:
                # Any failure during half-open probe trips back to OPEN immediately
                self._successful_probes = 0
                self._transition_to(self.STATE_OPEN)
            elif self._state == self.STATE_CLOSED:
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(self.STATE_OPEN)

    def execute(
        self,
        func: Callable[..., T],
        *args: Any,
        fallback: Optional[Callable[[Exception], T]] = None,
        **kwargs: Any,
    ) -> T:
        """Execute call through the circuit breaker, invoking fallback if provided."""
        current_state = self.state
        if current_state == self.STATE_OPEN:
            exc = CircuitBreakerOpenError(
                f"Circuit breaker '{self.name}' is OPEN. Outbound requests are blocked.",
                operation="circuit_breaker_call",
                entity=self.name,
                layer="resilience",
            )
            if fallback is not None:
                logger.info(f"[RESILIENCE] CircuitBreaker '{self.name}' OPEN. Invoking fallback.")
                return fallback(exc)
            raise exc

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as exc:
            self.record_failure(exc)
            if fallback is not None:
                logger.info(f"[RESILIENCE] CircuitBreaker '{self.name}' error caught. Invoking fallback.")
                return fallback(exc)
            raise


class Bulkhead:
    """Bulkhead concurrency limiter (§7) to prevent outbound socket exhaustion."""

    def __init__(self, name: str, max_concurrent_calls: int = 10):
        self.name = name
        self.max_concurrent_calls = max_concurrent_calls
        self._semaphore = threading.Semaphore(max_concurrent_calls)

    def __enter__(self):
        acquired = self._semaphore.acquire(blocking=False)
        if not acquired:
            raise ProviderUnavailableError(
                f"Bulkhead limit reached for '{self.name}'. Concurrency exceeded {self.max_concurrent_calls}.",
                operation="bulkhead_acquire",
                entity=self.name,
                layer="resilience",
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._semaphore.release()


def calculate_jittered_backoff(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Calculate randomized full jitter backoff: random.uniform(0, min(cap, base * 2 ** attempt))."""
    exponential = base * (2 ** attempt)
    sleep_limit = min(cap, exponential)
    return random.uniform(0, sleep_limit)
