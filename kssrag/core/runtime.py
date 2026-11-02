"""
Runtime utilities and lazy loader helpers for KSS RAG.
Conforms to house-standard.md §4 (explicit naming) and §9 (structured logger).
"""
import importlib
from typing import Tuple, Any, Optional
from .logging import logger

_FAISS_CACHE: dict = {"available": None, "avx_type": None}


def setup_faiss(vector_store_type: Optional[str] = None) -> Tuple[bool, str]:
    """Handle FAISS initialization lazily when explicitly requested."""
    if _FAISS_CACHE["available"] is not None:
        return _FAISS_CACHE["available"], _FAISS_CACHE["avx_type"]

    faiss_available = False
    faiss_avx_type = "not_loaded"

    if vector_store_type in ["faiss", "hybrid_online"]:
        try:
            faiss_import_attempts = [
                ("AVX512-SPR", "faiss.swigfaiss_avx512_spr"),
                ("AVX512", "faiss.swigfaiss_avx512"),
                ("AVX2", "faiss.swigfaiss_avx2"),
                ("Standard", "faiss.swigfaiss"),
            ]

            for avx_type, import_path in faiss_import_attempts:
                try:
                    logger.info(f"Loading faiss with {avx_type} support.")
                    faiss_module = importlib.import_module(import_path)
                    globals().update({name: getattr(faiss_module, name) for name in dir(faiss_module) if not name.startswith('_')})
                    faiss_available = True
                    faiss_avx_type = avx_type
                    logger.info(f"Successfully loaded faiss with {avx_type} support.")
                    break
                except ImportError as e:
                    logger.debug(f"Could not load library with {avx_type} support: {e}")
                    continue

            if not faiss_available:
                logger.warning("Could not load any FAISS version. FAISS-based vector stores will be disabled.")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {str(e)}")
            faiss_available = False
    else:
        logger.debug(f"Skipping FAISS initialization for vector store: {vector_store_type}")

    _FAISS_CACHE["available"] = faiss_available
    _FAISS_CACHE["avx_type"] = faiss_avx_type
    return faiss_available, faiss_avx_type


def validate_config() -> bool:
    """Validate active configuration."""
    from .config import config
    if not config.OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set. LLM functionality will not work.")
    return True


def kss_signature() -> str:
    """Return author attribution."""
    return "Built by Ksschkw (github.com/Ksschkw)"


def import_custom_component(import_path: str) -> Any:
    """Import a custom component from a string path."""
    try:
        module_path, class_name = import_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Failed to import custom component {import_path}: {str(e)}")
        raise
