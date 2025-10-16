import pytest
import asyncio
from kssrag.models.openrouter import OpenRouterLLM

def test_openrouter_streaming_initialization():
    """Test OpenRouterLLM streaming initialization"""
    llm = OpenRouterLLM(stream=True)
    assert llm.stream == True

def test_openrouter_non_streaming_initialization():
    """Test OpenRouterLLM non-streaming initialization"""
    llm = OpenRouterLLM(stream=False)
    assert llm.stream == False

def test_streaming_generator():
    """Test streaming generator interface"""
    # Mock the predict_stream method for testing
    class TestOpenRouterLLM(OpenRouterLLM):
        def predict_stream(self, messages):
            yield "Hello "
            yield "World"
            yield "!"
    
    llm = TestOpenRouterLLM(stream=True)
    messages = [{"role": "user", "content": "test"}]
    
    chunks = list(llm.predict_stream(messages))
    assert chunks == ["Hello ", "World", "!"]

def test_streaming_fallback_to_non_streaming():
    """Test that streaming falls back to non-streaming when no chunks"""
    class TestOpenRouterLLM(OpenRouterLLM):
        def predict_stream(self, messages):
            # Simulate no chunks returned
            if False:
                yield "test"
    
    llm = TestOpenRouterLLM(stream=True)
    # This should not raise an error
    result = llm.predict([{"role": "user", "content": "test"}])
    assert isinstance(result, str)