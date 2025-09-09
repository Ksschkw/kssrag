import logging
import importlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:KSSRAG:%(message)s'
)
logger = logging.getLogger("KSSRAG")


def setup_faiss():
    """Handle FAISS initialization with proper error handling and fallbacks"""
    faiss_available = False
    faiss_avx_type = "standard"
    
    # Only try to import FAISS if it's actually needed
    from ..config import config
    if config.VECTOR_STORE_TYPE in ["faiss", "hybrid_online"]:
        try:
            # Try different FAISS versions in order of preference
            faiss_import_attempts = [
                ("AVX512-SPR", "faiss.swigfaiss_avx512_spr"),
                ("AVX512", "faiss.swigfaiss_avx512"),
                ("AVX2", "faiss.swigfaiss_avx2"),
                ("Standard", "faiss.swigfaiss")
            ]
            
            for avx_type, import_path in faiss_import_attempts:
                try:
                    logger.info(f"Loading faiss with {avx_type} support.")
                    # Dynamic import
                    import importlib
                    faiss_module = importlib.import_module(import_path)
                    # Make the FAISS symbols available globally
                    globals().update({name: getattr(faiss_module, name) for name in dir(faiss_module) if not name.startswith('_')})
                    
                    faiss_available = True
                    faiss_avx_type = avx_type
                    logger.info(f"Successfully loaded faiss with {avx_type} support.")
                    break
                    
                except ImportError as e:
                    logger.info(f"Could not load library with {avx_type} support due to: {repr(e)}")
                    continue
                    
            if not faiss_available:
                logger.warning("Could not load any FAISS version. FAISS-based vector stores will be disabled.")
                
        except Exception as e:
            logger.error(f"Failed to initialize FAISS: {str(e)}")
            faiss_available = False
    
    return faiss_available, faiss_avx_type

# Initialize FAISS only when needed
FAISS_AVAILABLE, FAISS_AVX_TYPE = setup_faiss()

# Your signature in the code
def kss_signature():
    return "Built with HATE by Ksschkw (github.com/Ksschkw)"

def validate_config():
    """Validate the configuration"""
    from ..config import config
    
    if not config.OPENROUTER_API_KEY:
        logger.warning("OPENROUTER_API_KEY not set. LLM functionality will not work.")
    
    if config.VECTOR_STORE_TYPE in ["faiss", "hybrid_online"] and not FAISS_AVAILABLE:
        logger.warning(f"FAISS not available. Falling back to HYBRID_OFFLINE vector store.")
        config.VECTOR_STORE_TYPE = "hybrid_offline"
        
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
