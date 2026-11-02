"""
Strict integration tests for Outbound Resilience Primitives (§7).
Covers:
- CircuitBreaker states: CLOSED, OPEN, HALF_OPEN
- Fast failure during OPEN state
- Fallback activation during OPEN state
- Bulkhead concurrency limiting
- Jittered exponential backoff
"""
import time
import pytest

from kssrag.core.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    Bulkhead,
    calculate_jittered_backoff,
)
from kssrag.core.errors import ProviderUnavailableError


def test_circuit_breaker_lifecycle_and_state_transitions():
    """Verify circuit breaker CLOSED -> OPEN -> HALF_OPEN -> CLOSED state machine (§7)."""
    breaker = CircuitBreaker(
        name="test_llm_provider",
        failure_threshold=3,
        recovery_timeout_seconds=0.1,  # short timeout for hermetic test
        probe_count=2,
    )

    assert breaker.state == CircuitBreaker.STATE_CLOSED

    def failing_call():
        raise RuntimeError("Network timeout")

    # 1. Cause failures up to threshold
    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.execute(failing_call)
        assert breaker.state == CircuitBreaker.STATE_CLOSED

    # 3rd failure trips the breaker to OPEN
    with pytest.raises(RuntimeError):
        breaker.execute(failing_call)
    assert breaker.state == CircuitBreaker.STATE_OPEN

    # 2. While OPEN, call fails fast without executing the function
    call_executed = False

    def should_not_run():
        nonlocal call_executed
        call_executed = True
        return "ok"

    with pytest.raises(CircuitBreakerOpenError):
        breaker.execute(should_not_run)
    assert call_executed is False

    # 3. Fallback executes when breaker is OPEN
    fallback_result = breaker.execute(
        should_not_run,
        fallback=lambda exc: "fallback_response_typed",
    )
    assert fallback_result == "fallback_response_typed"

    # 4. Wait for recovery timeout to transition to HALF_OPEN
    time.sleep(0.12)
    assert breaker.state == CircuitBreaker.STATE_HALF_OPEN

    # 5. Successful probes in HALF_OPEN close the breaker
    probe1 = breaker.execute(lambda: "probe_success_1")
    assert probe1 == "probe_success_1"
    assert breaker.state == CircuitBreaker.STATE_HALF_OPEN  # needs 2 probes

    probe2 = breaker.execute(lambda: "probe_success_2")
    assert probe2 == "probe_success_2"
    assert breaker.state == CircuitBreaker.STATE_CLOSED


def test_circuit_breaker_half_open_failure_re_trips_to_open():
    """Verify that a single failure during HALF_OPEN immediately re-trips to OPEN (§7)."""
    breaker = CircuitBreaker(
        name="test_fragile_provider",
        failure_threshold=1,
        recovery_timeout_seconds=0.05,
        probe_count=2,
    )

    # Trip to OPEN
    with pytest.raises(ValueError):
        breaker.execute(lambda: (_ for _ in ()).throw(ValueError("err")))
    assert breaker.state == CircuitBreaker.STATE_OPEN

    time.sleep(0.06)
    assert breaker.state == CircuitBreaker.STATE_HALF_OPEN

    # Failure during probe immediately trips back to OPEN
    with pytest.raises(ValueError):
        breaker.execute(lambda: (_ for _ in ()).throw(ValueError("probe failed")))
    assert breaker.state == CircuitBreaker.STATE_OPEN


def test_bulkhead_concurrency_ceiling():
    """Verify Bulkhead rejects executions when concurrent capacity is exhausted (§7)."""
    bulkhead = Bulkhead("test_dependency", max_concurrent_calls=2)

    with bulkhead:
        with bulkhead:
            # 3rd concurrent entry exceeds ceiling
            with pytest.raises(ProviderUnavailableError) as exc_info:
                with bulkhead:
                    pass
            assert "Bulkhead limit reached" in str(exc_info.value)

    # After exit, semaphore is released and entry succeeds
    with bulkhead:
        pass


def test_jittered_backoff_bounds_and_randomness():
    """Verify calculate_jittered_backoff respects bounds and exhibits randomized jitter (§7)."""
    samples = [calculate_jittered_backoff(attempt=2, base=1.0, cap=10.0) for _ in range(50)]
    assert all(0.0 <= s <= 4.0 for s in samples)
    # Check that there is randomized variance (not deterministic)
    assert len(set(samples)) > 20
