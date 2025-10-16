import logging
import importlib

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:KSSRAG:%(message)s'
)
logger = logging.getLogger("KSSRAG")

# Initialize as None - will be set when actually needed
FAISS_AVAILABLE = None
FAISS_AVX_TYPE = None

def setup_faiss(vector_store_type: str = None):
    """Handle FAISS initialization - only when explicitly called"""
    global FAISS_AVAILABLE, FAISS_AVX_TYPE
    
    # If already initialized, return cached values
    if FAISS_AVAILABLE is not None:
        return FAISS_AVAILABLE, FAISS_AVX_TYPE
    
    faiss_available = False
    faiss_avx_type = "not_loaded"
    
    # Only load FAISS if explicitly using FAISS-based stores
    if vector_store_type in ["faiss", "hybrid_online"]:
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
                    faiss_module = importlib.import_module(import_path)
                    # Make the FAISS symbols available globally
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
        # Not using FAISS, don't load it
        logger.debug(f"Skipping FAISS initialization for vector store: {vector_store_type}")
    
    # Cache the results
    FAISS_AVAILABLE = faiss_available
    FAISS_AVX_TYPE = faiss_avx_type
    
    return faiss_available, faiss_avx_type

def validate_config():
    """Validate the configuration - don't auto-load FAISS here"""
    try:
        from ..config import config
        
        if not config.OPENROUTER_API_KEY:
            logger.warning("OPENROUTER_API_KEY not set. LLM functionality will not work.")
        
        # Don't auto-load FAISS here - let the vector stores handle it
        return True
    except ImportError:
        # Config not available, continue anyway
        return True

# Your signature in the code
def kss_signature():
    return "Built with HATE by Ksschkw (github.com/Ksschkw)"

def import_custom_component(import_path: str):
    """Import a custom component from a string path"""
    try:
        module_path, class_name = import_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except (ImportError, AttributeError, ValueError) as e:
        logger.error(f"Failed to import custom component {import_path}: {str(e)}")
        raise

# Remove the auto-initialization at module level
# FAISS will now only load when explicitly called by vector stores that need it