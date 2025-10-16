import pytest
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock
from kssrag.utils.ocr_loader import OCRLoader

def test_ocr_loader_initialization():
    """Test OCRLoader initialization with mocked PaddleOCR"""
    with patch('kssrag.utils.ocr_loader.PaddleOCR') as mock_paddle:
        mock_instance = Mock()
        mock_paddle.return_value = mock_instance
        
        loader = OCRLoader()
        assert loader.paddle_ocr == mock_instance

def test_ocr_loader_invalid_mode():
    """Test OCRLoader with invalid mode"""
    with patch('kssrag.utils.ocr_loader.PaddleOCR') as mock_paddle:
        mock_instance = Mock()
        mock_paddle.return_value = mock_instance
        
        loader = OCRLoader()
        
        with pytest.raises(ValueError, match="Invalid OCR mode"):
            loader.extract_text("test.jpg", "invalid_mode")

def test_ocr_loader_file_not_found():
    """Test OCRLoader with non-existent file"""
    with patch('kssrag.utils.ocr_loader.PaddleOCR') as mock_paddle:
        mock_instance = Mock()
        mock_paddle.return_value = mock_instance
        
        loader = OCRLoader()
        
        with pytest.raises(FileNotFoundError):
            loader.extract_text("nonexistent.jpg", "typed")

def test_ocr_loader_typed_mode():
    """Test OCRLoader typed mode"""
    with patch('kssrag.utils.ocr_loader.PaddleOCR') as mock_paddle:
        with patch('kssrag.utils.ocr_loader.pytesseract') as mock_tesseract:
            with patch('kssrag.utils.ocr_loader.Image') as mock_image:
                with patch('kssrag.utils.ocr_loader.os.path.exists') as mock_exists:
                    # Mock file exists
                    mock_exists.return_value = True
                    
                    # Mock image opening
                    mock_img_instance = MagicMock()
                    mock_image.open.return_value = mock_img_instance
                    
                    # Mock OCR result
                    mock_tesseract.image_to_string.return_value = "Typed text content"
                    
                    mock_paddle_instance = Mock()
                    mock_paddle.return_value = mock_paddle_instance
                    
                    loader = OCRLoader()
                    
                    result = loader.extract_text("test.jpg", "typed")
                    assert result == "Typed text content"
                    mock_tesseract.image_to_string.assert_called_once_with(mock_img_instance)

def test_ocr_loader_handwritten_mode():
    """Test OCRLoader handwritten mode"""
    with patch('kssrag.utils.ocr_loader.PaddleOCR') as mock_paddle:
        with patch('kssrag.utils.ocr_loader.cv2') as mock_cv2:
            with patch('kssrag.utils.ocr_loader.os.path.exists') as mock_exists:
                # Mock file exists
                mock_exists.return_value = True
                
                # Mock image reading
                mock_cv2.imread.return_value = "mock_image"
                
                # Mock OCR result
                mock_paddle_instance = Mock()
                mock_paddle_instance.ocr.return_value = [[[None, ["Handwritten text", 0.9]]]]
                mock_paddle.return_value = mock_paddle_instance
                
                loader = OCRLoader()
                
                result = loader.extract_text("test.jpg", "handwritten")
                assert result == "Handwritten text"
                mock_paddle_instance.ocr.assert_called_once_with("mock_image", cls=True)

def test_ocr_loader_paddle_not_initialized():
    """Test OCRLoader when PaddleOCR is not initialized"""
    with patch('kssrag.utils.ocr_loader.PaddleOCR') as mock_paddle:
        mock_paddle.return_value = None  # Simulate initialization failure
        
        loader = OCRLoader()
        loader.paddle_ocr = None  # Force the failure state
        
        with pytest.raises(RuntimeError, match="PaddleOCR not initialized"):
            loader.extract_text("test.jpg", "handwritten")

def test_ocr_loader_empty_text():
    """Test OCRLoader when no text is extracted"""
    with patch('kssrag.utils.ocr_loader.PaddleOCR') as mock_paddle:
        with patch('kssrag.utils.ocr_loader.pytesseract') as mock_tesseract:
            with patch('kssrag.utils.ocr_loader.Image') as mock_image:
                with patch('kssrag.utils.ocr_loader.os.path.exists') as mock_exists:
                    # Mock file exists
                    mock_exists.return_value = True
                    
                    # Mock image opening
                    mock_img_instance = MagicMock()
                    mock_image.open.return_value = mock_img_instance
                    
                    # Mock empty OCR result
                    mock_tesseract.image_to_string.return_value = "   "  # Only whitespace
                    
                    mock_paddle_instance = Mock()
                    mock_paddle.return_value = mock_paddle_instance
                    
                    loader = OCRLoader()
                    
                    result = loader.extract_text("test.jpg", "typed")
                    assert result == ""  # Should return empty string

@pytest.mark.skipif(not os.getenv('TEST_OCR'), reason="OCR tests require actual OCR dependencies")
def test_ocr_loader_integration():
    """Integration test for OCRLoader with actual image"""
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        temp_file = f.name
    
    try:
        # Create a simple test image with text
        from PIL import Image, ImageDraw
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((50, 40), "OCR Test Text", fill='black')
        img.save(temp_file)
        
        loader = OCRLoader()
        text = loader.extract_text(temp_file, "typed")
        
        assert isinstance(text, str)
        # Note: Actual OCR text recognition might vary
        
    finally:
        if os.path.exists(temp_file):
            os.unlink(temp_file)