import pytest
from kssrag.core.chunkers import ImageChunker, OCR_AVAILABLE

@pytest.mark.skipif(not OCR_AVAILABLE, reason="OCR dependencies not available")
def test_image_chunker_basic():
    """Basic test for ImageChunker - just check it initializes"""
    chunker = ImageChunker(ocr_mode="typed")
    assert chunker.ocr_mode == "typed"

@pytest.mark.skipif(not OCR_AVAILABLE, reason="OCR dependencies not available") 
def test_image_chunker_modes():
    """Test that ImageChunker accepts valid modes"""
    chunker_typed = ImageChunker(ocr_mode="typed")
    chunker_handwritten = ImageChunker(ocr_mode="handwritten")
    
    assert chunker_typed.ocr_mode == "typed"
    assert chunker_handwritten.ocr_mode == "handwritten"