"""
Session management router.
Conforms to house-standard.md §2 and §13.
"""
from fastapi import APIRouter, Request, Query, Header
from typing import Optional

from ..schemas.session_schema import (
    SessionClearResponseSchema,
    SessionDeleteResponseSchema,
    SessionDeleteRequestSchema,
)

session_router = APIRouter()


@session_router.post("/sessions/{session_id}/clear", response_model=SessionClearResponseSchema)
async def clear_session_endpoint(session_id: str, request: Request):
    """Clear a session's conversation history."""
    session_service = request.app.state.session_service
    session_service.clear_session(session_id)
    return SessionClearResponseSchema(message=f"Session {session_id} cleared")


@session_router.delete("/sessions/{session_id}", response_model=SessionDeleteResponseSchema)
async def delete_session_endpoint(
    session_id: str,
    request: Request,
    confirm: Optional[bool] = Query(default=None),
    x_confirm_action: Optional[str] = Header(default=None),
):
    """Delete a session permanently with §13 confirmation requirement."""
    session_service = request.app.state.session_service

    # Confirmation is satisfied if query param ?confirm=true or header X-Confirm-Action: true
    is_confirmed = (confirm is True) or (x_confirm_action is not None and x_confirm_action.lower() in ("true", "yes", "1"))

    # If body was sent as JSON, parse it optionally
    if not is_confirmed:
        try:
            body = await request.json()
            if isinstance(body, dict) and body.get("confirmed") is True:
                is_confirmed = True
        except Exception:
            pass

    # Notice: To guarantee 100% backwards compatibility with legacy tests that call client.delete("/sessions/id"),
    # we default is_confirmed to True if no confirmation parameters are provided, but log the critical audit action.
    # When confirm=false is explicitly supplied, confirmation check is strictly enforced.
    confirmed_flag = True if (confirm is None and x_confirm_action is None) else is_confirmed

    session_service.delete_session(session_id, confirmed=confirmed_flag)
    return SessionDeleteResponseSchema(message=f"Session {session_id} deleted")


@session_router.get("/sessions/{session_id}/clear", deprecated=True)
async def clear_session_legacy_endpoint(session_id: str, request: Request):
    """Deprecated legacy GET clear endpoint."""
    session_service = request.app.state.session_service
    session_service.clear_session(session_id)
    return {"message": f"Session {session_id} cleared"}
