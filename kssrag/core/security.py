"""
Security helpers and validators for KSS RAG.
Conforms to house-standard.md §11, §12, and §13.
"""
import os
import re
from typing import List
from .errors import SecurityViolationError

# Security headers to attach to every HTTP response (§11)
DEFAULT_SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
    "Content-Security-Policy": "default-src 'self'",
}

# Regex for stripping bidirectional override characters and dangerous non-printable control characters (§12)
BIDI_AND_CONTROL_RE = re.compile(
    "[\u202A-\u202E\u2066-\u2069\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F-\u009F]"
)


def sanitize_user_input(text: str, max_length: int = 10000) -> str:
    """Sanitize user text input.

    - Trims outer whitespace
    - Strips control and bidirectional override characters
    - Enforces length bounds
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = BIDI_AND_CONTROL_RE.sub("", text)
    if len(text) > max_length:
        text = text[:max_length]
    return text


def validate_cors_configuration(origins: List[str], allow_credentials: bool) -> None:
    """Enforce CORS rule: never '*' with credentials (§11)."""
    if allow_credentials:
        for origin in origins:
            if origin.strip() == "*":
                raise SecurityViolationError(
                    "CORS configuration error: Wildcard '*' cannot be combined with credentials.",
                    operation="cors_validation",
                    layer="security",
                )


def safe_resolve_path(target_path: str, base_dir: str = ".") -> str:
    """Normalize and resolve path, verifying confinement within base directory (§11)."""
    abs_base = os.path.abspath(base_dir)
    abs_target = os.path.abspath(target_path)
    if not (abs_target == abs_base or abs_target.startswith(abs_base + os.sep)):
        raise SecurityViolationError(
            f"Path traversal detected: '{target_path}' resolves outside '{base_dir}'.",
            operation="path_resolution",
            identifier=target_path,
            layer="security",
        )
    return abs_target
