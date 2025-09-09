"""
Local LLM implementations for offline usage
"""
from typing import List, Dict, Any
from ..utils.helpers import logger

class LocalLLM:
    """Base class for local LLM implementations"""
    
    def predict(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using a local model"""
        raise NotImplementedError("Subclasses must implement this method")

class MockLLM(LocalLLM):
    """Mock LLM for testing without API calls"""
    
    def predict(self, messages: List[Dict[str, str]]) -> str:
        """Generate a mock response for testing"""
        logger.info("Using mock LLM for response generation")
        
        # Extract the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break
        
        return f"This is a mock response to: {user_message}"

# TO-DO - Add more local LLM implementations as needed - I hope i do not forget, someone if you are seeing this, the date i added this was 8th of september 2025, how long has it been? DM me on WhatsApp +2349019549473.