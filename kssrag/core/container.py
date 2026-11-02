"""
Composition root for KSS RAG dependency injection.
Conforms to house-standard.md §2 (Hard Rule 5) and §17.
"""
from typing import Optional, Dict, Any

from .config import Config, config as default_config
from .logging import get_logger, logger
from .resilience import CircuitBreaker, Bulkhead


class Container:
    """Dependency injection composition root."""

    def __init__(self, config_instance: Optional[Config] = None):
        self.config = config_instance or default_config
        self.logger = get_logger("KSSRAG")
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._bulkheads: Dict[str, Bulkhead] = {}

    def get_circuit_breaker(
        self,
        name: str = "outbound_llm",
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
    ) -> CircuitBreaker:
        """Return or create a dependency circuit breaker."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(
                name=name,
                failure_threshold=failure_threshold,
                recovery_timeout_seconds=recovery_timeout_seconds,
            )
        return self._circuit_breakers[name]

    def get_bulkhead(self, name: str = "outbound_llm", max_concurrent: int = 10) -> Bulkhead:
        """Return or create a concurrency bulkhead."""
        if name not in self._bulkheads:
            self._bulkheads[name] = Bulkhead(name=name, max_concurrent_calls=max_concurrent)
        return self._bulkheads[name]
