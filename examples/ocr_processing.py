#!/usr/bin/env python3
"""
OCR processing example showcasing handwritten and typed text recognition
"""
import os
from kssrag import KSSRAG, Config

def main():
    # Configuration optimized for OCR processing
    config = Config(
        OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY"),
        OCR_DEFAULT_MODE="typed",  # Can be "typed" or "handwritten"
        CHUNK_SIZE=600,
        TOP_K=5
    )
    
    rag = KSSRAG(config=config)
    
    # Process different types of images
    images_to_process = [
        ("typed_document.jpg", "typed"),
        ("handwritten_notes.jpg", "handwritten"),
        ("mixed_content.png", "typed")  # Default to typed for mixed content
    ]
    
    for image_path, ocr_mode in images_to_process:
        try:
            print(f"Processing {image_path} with {ocr_mode} OCR...")
            
            # Update OCR mode for this specific image
            config.OCR_DEFAULT_MODE = ocr_mode
            
            # Load and process image
            rag.load_document(image_path, format="image")
            print(f"✓ Successfully processed {image_path}")
            
        except Exception as e:
            print(f"✗ Failed to process {image_path}: {str(e)}")
    
    # Query across all processed images
    print("\nQuerying processed documents...")
    response = rag.query("Extract and summarize all important information from the images")
    print(f"Summary: {response}")

if __name__ == "__main__":
    main()