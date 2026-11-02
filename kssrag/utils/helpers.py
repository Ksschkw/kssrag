"""
Backward-compatibility facade for helpers.
Canonical implementations live in kssrag.core.runtime and kssrag.core.logging.
"""
from ..core.logging import logger
from ..core.runtime import (
    setup_faiss,
    validate_config,
    kss_signature,
    import_custom_component,
    _FAISS_CACHE,
)

FAISS_AVAILABLE = _FAISS_CACHE["available"]
FAISS_AVX_TYPE = _FAISS_CACHE["avx_type"]

__all__ = [
    "logger",
    "setup_faiss",
    "validate_config",
    "kss_signature",
    "import_custom_component",
    "FAISS_AVAILABLE",
    "FAISS_AVX_TYPE",
]
