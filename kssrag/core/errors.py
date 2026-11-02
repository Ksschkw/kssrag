"""
Unified error hierarchy and correlation ID tracking for KSS RAG.
Conforms to house-standard.md §9.
"""
import uuid
from contextvars import ContextVar
from typing import Optional, Dict, Any

# ContextVar for threading correlation ID across asynchronous and synchronous tasks
correlation_id_ctx: ContextVar[str] = ContextVar("correlation_id", default="")


def get_correlation_id() -> str:
    """Return the active correlation ID or generate a new one if not present."""
    cid = correlation_id_ctx.get()
    if not cid:
        cid = str(uuid.uuid4())
        correlation_id_ctx.set(cid)
    return cid


def set_correlation_id(cid: str) -> None:
    """Explicitly set the active correlation ID."""
    correlation_id_ctx.set(cid)


class KSSRAGError(Exception):
    """Base error for all internal domain errors in KSS RAG."""

    def __init__(
        self,
        message: str,
        operation: str = "unknown_operation",
        entity: Optional[str] = None,
        identifier: Optional[str] = None,
        layer: str = "core",
        cause: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.operation = operation
        self.entity = entity
        self.identifier = identifier
        self.layer = layer
        self.cause = cause
        self.details = details or {}
        self.correlation_id = get_correlation_id()

    def to_internal_dict(self) -> Dict[str, Any]:
        """Internal view for logs and engineers with full diagnostic context."""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "operation": self.operation,
            "entity": self.entity,
            "identifier": self.identifier,
            "layer": self.layer,
            "correlation_id": self.correlation_id,
            "cause": str(self.cause) if self.cause else None,
            "details": self.details,
        }

    def to_external_response(self) -> Dict[str, Any]:
        """External view safe for clients with four-part guidance (§10)."""
        return {
            "error": {
                "message": "We could not complete your request. Please check your query and try again.",
                "action": "Verify your input and retry the operation.",
                "reference": self.correlation_id,
            }
        }


class NotFoundError(KSSRAGError):
    """Raised when an entity, document, or session is not found."""

    def to_external_response(self) -> Dict[str, Any]:
        entity_name = self.entity or "resource"
        return {
            "error": {
                "message": f"The requested {entity_name} could not be located.",
                "action": f"Confirm the {entity_name} identifier and try again.",
                "reference": self.correlation_id,
            }
        }


class ConflictError(KSSRAGError):
    """Raised on state conflict or concurrent mutation."""

    def to_external_response(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": "The resource has changed or an operation is already in progress.",
                "action": "Refresh your view and retry your submission.",
                "reference": self.correlation_id,
            }
        }


class ValidationError(KSSRAGError):
    """Raised when input validation fails at domain or transport boundary."""

    def to_external_response(self) -> Dict[str, Any]:
        field = self.details.get("field", "input")
        fix = self.details.get("fix", "Please provide a valid value.")
        return {
            "error": {
                "message": f"The {field} provided is not acceptable. {fix}",
                "action": f"Correct the {field} and submit again.",
                "reference": self.correlation_id,
            }
        }


class ProviderUnavailableError(KSSRAGError):
    """Raised when an outbound dependency (e.g. LLM provider) is down or circuit broken."""

    def to_external_response(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": "The language model service is temporarily unreachable.",
                "action": "Please wait a moment while we reconnect, then retry your query.",
                "reference": self.correlation_id,
            }
        }


class RateLimitExceededError(KSSRAGError):
    """Raised when a request exceeds configured request frequency."""

    def to_external_response(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": "You have submitted too many requests in a short period.",
                "action": "Please pause for a few seconds before submitting another request.",
                "reference": self.correlation_id,
            }
        }


class SecurityViolationError(KSSRAGError):
    """Raised on security violation (unauthorized access, path traversal, forbidden origin)."""

    def to_external_response(self) -> Dict[str, Any]:
        return {
            "error": {
                "message": "This request was rejected for security considerations.",
                "action": "Check your authorization credentials and request format.",
                "reference": self.correlation_id,
            }
        }


class LLMError(RuntimeError):
    """Raised when an LLM adapter cannot produce a response because every model failed.

    Defined here rather than in ``models/base`` so that ``core`` never has to
    import from ``models``: providers depend on core, never the reverse.
    ``models.base`` re-exports it, so ``from kssrag.models.base import LLMError``
    keeps working.
    """


class CriticalActionConfirmationRequiredError(KSSRAGError):
    """Raised when a destructive action was attempted without required confirmation (§13)."""

    def to_external_response(self) -> Dict[str, Any]:
        action_name = self.details.get("action_name", "perform this destructive action")
        consequence = self.details.get("consequence", "This will permanently remove the data.")
        return {
            "error": {
                "message": f"Confirmation required to {action_name}. {consequence}",
                "action": "Send the request with confirmed=true or X-Confirm-Action header.",
                "reference": self.correlation_id,
            }
        }
