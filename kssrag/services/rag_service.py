"""
RAG orchestration service.

Conforms to house-standard.md section 2 (business logic) and section 9 (typed
domain errors). The retriever, the language model, and the circuit breaker are
injected by the composition root; nothing here constructs them.
"""
from typing import Any, Dict, Generator, Optional

from ..core.errors import ProviderUnavailableError
from ..core.logging import logger
from ..core.resilience import CircuitBreaker
from ..crud.session_crud import SessionCrud
from ..models.entities.query_model import QueryResultModel


class RAGService:
    """Answer queries against the indexed knowledge base."""

    def __init__(self, session_crud: SessionCrud, circuit_breaker: CircuitBreaker):
        self.session_crud = session_crud
        self.circuit_breaker = circuit_breaker

    def _agent_for(self, session_id: str):
        """The agent that owns this session's conversation state."""
        agent = self.session_crud.get_by_id(session_id)
        if agent is None:
            agent = self.session_crud.create(session_id)
        return agent

    def execute_query(
        self,
        query: str,
        session_id: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> QueryResultModel:
        """Answer ``query`` within ``session_id``, guarded by the circuit breaker.

        Raises ProviderUnavailableError when the breaker is open or the agent
        raises.
        """
        agent = self._agent_for(session_id)

        def call_agent() -> str:
            return agent.query(query, top_k=top_k, metadata_filter=metadata_filter)

        def fallback(exc: Exception) -> str:
            logger.warning(f"RAGService query fallback activated: {exc}")
            raise ProviderUnavailableError(
                "The language model service is unavailable.",
                operation="rag_query",
                entity="session",
                identifier=session_id,
                layer="service",
                cause=exc,
            )

        response_text = self.circuit_breaker.execute(call_agent, fallback=fallback)
        return QueryResultModel(
            query=query,
            response=response_text,
            session_id=session_id,
        )

    def execute_stream(
        self,
        query: str,
        top_k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Generator[str, None, None]:
        """Yield answer tokens.

        Streaming is not session scoped: it uses the process-wide agent, which
        matches the behaviour of the previous implementation.
        """
        for token in self.session_crud.default_agent.query_stream(
            query, top_k=top_k, metadata_filter=metadata_filter
        ):
            yield token
