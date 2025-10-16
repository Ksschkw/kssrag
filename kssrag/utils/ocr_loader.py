import os
import cv2
import pytesseract
from paddleocr import PaddleOCR
from pathlib import Path
from PIL import Image
from .helpers import logger

class OCRLoader:
    """Production OCR handler with PaddleOCR (handwritten) and Tesseract (typed)"""
    
    def __init__(self):
        self.paddle_ocr = None
        self._initialize_paddle_ocr()
    
    # def _initialize_paddle_ocr(self):
    #     """Initialize PaddleOCR with custom model directories and fallback"""
    #     try:
    #         # Try to use custom model directories first
    #         det_model_dir = str(Path(__file__).parent.parent / 'paddle_models' / 'models' / 'ppocrv5_server_det')
    #         rec_model_dir = str(Path(__file__).parent.parent / 'paddle_models' / 'models' / 'ppocrv5_server_rec')
            
    #         # Create directories if they don't exist
    #         os.makedirs(det_model_dir, exist_ok=True)
    #         os.makedirs(rec_model_dir, exist_ok=True)
            
    #         # Try to initialize with custom directories
    #         try:
    #             self.paddle_ocr = PaddleOCR(
    #                 det_model_dir=det_model_dir,
    #                 rec_model_dir=rec_model_dir,
    #                 use_angle_cls=True,
    #                 lang="en"
    #             )
    #             logger.info("PaddleOCR initialized successfully with custom model directories")
                
    #         except (PermissionError, OSError) as e:
    #             logger.warning(f"Failed to initialize PaddleOCR with custom directories: {str(e)}. Using default directories.")
    #             # Fallback to default initialization
    #             self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
    #             logger.info("PaddleOCR initialized successfully with default directories")
                
    #     except Exception as e:
    #         logger.error(f"PaddleOCR initialization failed: {str(e)}")
    #         # Don't raise here - allow the loader to be created but OCR will fail when used
    #         self.paddle_ocr = None

    def _initialize_paddle_ocr(self):
        """Initialize PaddleOCR with better directory handling"""
        try:
            # Try to use custom model directories first
            det_model_dir = str(Path(__file__).parent.parent / 'paddle_models' / 'models' / 'ppocrv5_server_det')
            rec_model_dir = str(Path(__file__).parent.parent / 'paddle_models' / 'models' / 'ppocrv5_server_rec')
            
            # Create directories if they don't exist
            os.makedirs(det_model_dir, exist_ok=True)
            os.makedirs(rec_model_dir, exist_ok=True)
            
            # Check if custom directories have the required files
            custom_dirs_valid = (
                os.path.exists(det_model_dir) and 
                os.path.exists(rec_model_dir) and
                os.path.exists(os.path.join(det_model_dir, 'inference.yml')) and
                os.path.exists(os.path.join(rec_model_dir, 'inference.yml'))
            )
            
            if custom_dirs_valid:
                self.paddle_ocr = PaddleOCR(
                    det_model_dir=det_model_dir,
                    rec_model_dir=rec_model_dir,
                    use_angle_cls=True,
                    lang="en"
                )
                logger.info("PaddleOCR initialized successfully with custom model directories")
            else:
                logger.info("Custom model directories not found, using default PaddleOCR initialization")
                self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
                logger.info("PaddleOCR initialized successfully with default directories")
                
        except Exception as e:
            logger.warning(f"PaddleOCR initialization failed: {str(e)}. Using default initialization.")
            # Fallback to default initialization
            self.paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en")
    
    def ocr_tesseract(self, image_path: str) -> str:
        """OCR for typed text using Tesseract with error handling"""
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            
            if not text.strip():
                logger.warning(f"Tesseract extracted no text from {image_path}")
                
            return text.strip()
            
        except FileNotFoundError:
            # Re-raise FileNotFoundError directly
            raise
        except Exception as e:
            logger.error(f"Tesseract OCR failed for {image_path}: {str(e)}")
            raise RuntimeError(f"Tesseract OCR failed: {str(e)}")
    
    def ocr_paddle(self, image_path: str) -> str:
        """OCR for handwritten text using PaddleOCR with error handling"""
        if self.paddle_ocr is None:
            raise RuntimeError("PaddleOCR not initialized. OCR functionality unavailable.")
        
        try:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Could not read image at {image_path}")
            
            result = self.paddle_ocr.ocr(img, cls=True)
            lines = []
            
            if result and result[0]:
                for line in result[0]:
                    if line and len(line) >= 2:
                        text_content = line[1][0] if isinstance(line[1], (list, tuple)) and len(line[1]) > 0 else ""
                        if text_content:
                            lines.append(text_content)
            
            extracted_text = " ".join(lines).strip()
            
            if not extracted_text:
                logger.warning(f"PaddleOCR extracted no text from {image_path}")
                
            return extracted_text
            
        except FileNotFoundError:
            # Re-raise FileNotFoundError directly
            raise
        except Exception as e:
            logger.error(f"PaddleOCR failed for {image_path}: {str(e)}")
            raise RuntimeError(f"PaddleOCR failed: {str(e)}")
    
    def extract_text(self, image_path: str, mode: str = "typed") -> str:
        """Extract text from image using specified OCR engine"""
        if mode not in ["typed", "handwritten"]:
            raise ValueError(f"Invalid OCR mode: {mode}. Must be 'typed' or 'handwritten'")
        
        if mode == "handwritten":
            return self.ocr_paddle(image_path)
        else:  # typed
            return self.ocr_tesseract(image_path)