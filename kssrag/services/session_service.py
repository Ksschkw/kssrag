"""
Session lifecycle service.

Conforms to house-standard.md section 2 (business logic) and section 13
(critical action confirmation and audit).

This layer owns policy: whether a session exists, whether deletion was
confirmed, and what gets written to the audit log. Storage is delegated to
SessionCrud.
"""
from ..core.errors import CriticalActionConfirmationRequiredError, NotFoundError
from ..core.logging import logger
from ..crud.session_crud import SessionCrud


class SessionService:
    """Conversation session lifecycle, confirmation policy, and audit."""

    def __init__(self, session_crud: SessionCrud):
        self.session_crud = session_crud

    def clear_session(self, session_id: str) -> bool:
        """Reset a session's conversation history.

        Returns True. Raises NotFoundError when the session was never created.
        """
        agent = self.session_crud.get_by_id(session_id)
        if agent is None:
            raise NotFoundError(
                f"Session '{session_id}' not found.",
                operation="clear_session",
                entity="session",
                identifier=session_id,
                layer="service",
            )
        agent.clear_conversation()
        logger.info(f"[AUDIT] Cleared session history identifier={session_id}")
        return True

    def delete_session(self, session_id: str, confirmed: bool = False, actor: str = "client") -> bool:
        """Delete a session permanently.

        Raises CriticalActionConfirmationRequiredError when ``confirmed`` is
        false, NotFoundError when the session does not exist, and returns True
        on success. Every successful deletion writes an audit line.
        """
        if not confirmed:
            raise CriticalActionConfirmationRequiredError(
                "Confirmation required to delete session.",
                operation="delete_session",
                entity="session",
                identifier=session_id,
                layer="service",
                details={
                    "action_name": f"delete conversation session '{session_id}'",
                    "consequence": (
                        "All conversational memory for this session will be permanently erased."
                    ),
                },
            )

        if not self.session_crud.delete(session_id):
            raise NotFoundError(
                f"Session '{session_id}' does not exist.",
                operation="delete_session",
                entity="session",
                identifier=session_id,
                layer="service",
            )

        logger.info(
            f"[AUDIT] CRITICAL_ACTION delete_session actor={actor} "
            f"session_id={session_id} success=True"
        )
        return True
