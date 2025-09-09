import logging
import importlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:KSSRAG:%(message)s'
)
logger = logging.getLogger("KSSRAG")

# Your signature in the code
def kss_signature():
    return "Built with HATE by Ksschkw (github.com/Ksschkw)"

def validate_config():
    """Validate the configuration"""
    from ..config import config
    
    if not config.OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set. LLM functionality will not work.")
    
    return True

def import_custom_component(import_path: str):
    """Import a custom component from a string path"""
    try:
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Failed to import custom component {import_path}: {str(e)}")
        raise

# import os
# import logging
# from .utils.helpers import logger

# def setup_faiss():
#     """Handle FAISS initialization with proper error handling"""
#     try:
#         # Try to load with AVX2 support first
#         logger.info("Loading faiss with AVX2 support.")
#         from faiss.swigfaiss_avx2 import *
#         logger.info("Successfully loaded faiss with AVX2 support.")
#         return True
#     except ImportError as e:
#         logger.info(f"Could not load library with AVX2 support due to:\n{repr(e)}")
#         logger.info("Falling back to standard FAISS without AVX2 support")
#         try:
#             from faiss.swigfaiss import *
#             logger.info("Successfully loaded standard faiss.")
#             return False
#         except ImportError as e:
#             logger.error(f"Failed to load any FAISS version: {repr(e)}")
#             raise