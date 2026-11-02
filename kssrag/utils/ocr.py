"""
OCR utilities for KSS RAG.
Requires extra dependencies: `paddleocr`, `paddlepaddle`, `pytesseract`, `Pillow`.
Install via: pip install kssrag[ocr]
"""

try:
    import pytesseract
    from paddleocr import PaddleOCR
    from PIL import Image
except ImportError as e:
    raise ImportError(
        "OCR functionality requires extra dependencies. "
        "Install with: pip install kssrag[ocr]"
    ) from e

# Lazy initialization of PaddleOCR to avoid module-level singleton (§2)
_paddle_ocr_instance = None

def _get_paddle_ocr():
    global _paddle_ocr_instance
    if _paddle_ocr_instance is None:
        _paddle_ocr_instance = PaddleOCR(use_angle_cls=True, lang="en")
    return _paddle_ocr_instance


def ocr_tesseract(image_path: str) -> str:
    """OCR for typed text using Tesseract."""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text.strip()


def ocr_paddle(image_path: str) -> str:
    """OCR for handwritten text using PaddleOCR."""
    ocr_engine = _get_paddle_ocr()
    results = ocr_engine.ocr(image_path, cls=True)
    text = ""
    for line in results:
        for _, (txt, _) in line:
            text += txt + " "
    return text.strip()


def extract_text_from_image(image_path: str, mode: str = "typed") -> str:
    """
    Dispatch OCR engine.
    mode = 'typed' (Tesseract) or 'handwritten' (PaddleOCR).
    """
    if mode == "handwritten":
        return ocr_paddle(image_path)
    elif mode == "typed":
        return ocr_tesseract(image_path)
    else:
        raise ValueError("Invalid OCR mode. Choose 'typed' or 'handwritten'.")
