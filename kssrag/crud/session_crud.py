"""
Session storage.

Conforms to house-standard.md section 2: one entity, persistence only.

A conversation session is stateful: the agent that serves it owns the message
history and the rolling summaries. Storing the agent is what makes session
isolation real, so this module keeps a bounded, thread-safe LRU of live agents
keyed by session identifier. It performs no policy: no confirmation, no audit,
no error rendering. Those belong to the service layer.
"""
import threading
from collections import OrderedDict
from typing import Callable, List, Optional

from ..core.agents import RAGAgent


class SessionCrud:
    """Thread-safe bounded LRU of live conversation agents."""

    def __init__(self, agent_factory: Callable[[], RAGAgent], max_sessions: int = 1000):
        """Create a store that builds agents with ``agent_factory``.

        ``max_sessions`` bounds memory: past it the least recently used session
        is evicted, exactly as the previous adapter did. ``default_agent`` is
        the agent used for streaming requests, which are not session-scoped.
        """
        self.agent_factory = agent_factory
        self.max_sessions = max_sessions
        self.default_agent = agent_factory()
        self._sessions: "OrderedDict[str, RAGAgent]" = OrderedDict()
        self._lock = threading.Lock()

    def get_by_id(self, session_id: str) -> Optional[RAGAgent]:
        """Return the agent for ``session_id`` and mark it recently used."""
        with self._lock:
            agent = self._sessions.get(session_id)
            if agent is not None:
                self._sessions.move_to_end(session_id)
            return agent

    def create(self, session_id: str) -> RAGAgent:
        """Store and return a new agent, or the existing one if already present."""
        with self._lock:
            existing = self._sessions.get(session_id)
            if existing is not None:
                self._sessions.move_to_end(session_id)
                return existing

            agent = self.agent_factory()
            self._sessions[session_id] = agent
            while len(self._sessions) > self.max_sessions:
                self._sessions.popitem(last=False)
            return agent

    def delete(self, session_id: str) -> bool:
        """Remove the session, reporting whether it existed."""
        with self._lock:
            return self._sessions.pop(session_id, None) is not None

    def list(self) -> List[str]:
        """Session identifiers currently held, least recently used first."""
        with self._lock:
            return list(self._sessions.keys())
