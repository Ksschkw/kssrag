"""
Structured logging with secret redaction and emoji hygiene for KSS RAG.
Conforms to house-standard.md §8 and §9.
"""
import logging
import re
from typing import Any
from .errors import get_correlation_id

# Regex patterns for sensitive credentials to redact in logs
REDACTION_PATTERNS = [
    (re.compile(r'(?i)(bearer\s+)[a-zA-Z0-9_\-\.]{10,}'), r'\1[REDACTED_TOKEN]'),
    (re.compile(r'(?i)(api[_-]?key["\']?\s*[:=]\s*["\']?)[a-zA-Z0-9_\-\.]{8,}'), r'\1[REDACTED_KEY]'),
    (re.compile(r'(?i)(sk-[a-zA-Z0-9_\-]{15,})'), r'[REDACTED_KEY]'),
    (re.compile(r'(?i)(password["\']?\s*[:=]\s*["\']?)[^"\'\s]+'), r'\1[REDACTED_PASSWORD]'),
    (re.compile(r'(?i)(authorization["\']?\s*[:=]\s*["\']?)[^"\'\s]+'), r'\1[REDACTED_AUTH]'),
]

# Regex matching emojis / decorative non-ASCII characters to preserve §8 no-emoji hygiene
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F700-\U0001F77F"  # alchemical
    "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
    "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U0001FA00-\U0001FA6F"  # Chess Symbols
    "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
    "\U00002702-\U000027B0"  # Dingbats
    "\U000024C2-\U0001F251"
    "\u2713\u2714\u2717\u2718"  # checkmarks / crosses
    "]+",
    flags=re.UNICODE,
)


def redact_sensitive_data(message: str) -> str:
    """Sanitize message by masking API keys, tokens, and credentials."""
    for pattern, replacement in REDACTION_PATTERNS:
        message = pattern.sub(replacement, message)
    # Strip emojis in log output
    message = EMOJI_PATTERN.sub("", message)
    return message


class RedactingFormatter(logging.Formatter):
    """Formatter that injects the correlation ID and redacts secrets."""

    def format(self, record: logging.LogRecord) -> str:
        # Attach current correlation ID if not set
        if not hasattr(record, "correlation_id") or not record.correlation_id:
            record.correlation_id = get_correlation_id()

        # Sanitize message content
        if isinstance(record.msg, str):
            record.msg = redact_sensitive_data(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {k: redact_sensitive_data(str(v)) if isinstance(v, str) else v for k, v in record.args.items()}
            elif isinstance(record.args, tuple):
                record.args = tuple(redact_sensitive_data(str(v)) if isinstance(v, str) else v for v in record.args)

        formatted = super().format(record)
        return redact_sensitive_data(formatted)


def get_logger(name: str = "KSSRAG") -> logging.Logger:
    """Return a logger with redaction and correlation ID formatting."""
    log = logging.getLogger(name)
    if not log.handlers:
        handler = logging.StreamHandler()
        formatter = RedactingFormatter("[%(levelname)s] [%(correlation_id)s] %(name)s: %(message)s")
        handler.setFormatter(formatter)
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False
    return log


# Canonical default logger instance
logger = get_logger("KSSRAG")
