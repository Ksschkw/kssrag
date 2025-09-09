import requests
import json
from typing import List, Dict, Any, Optional
from ..utils.helpers import logger
from ..config import config

class OpenRouterLLM:
    """OpenRouter LLM interface with fallback models"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                 fallback_models: Optional[List[str]] = None):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.model = model or config.DEFAULT_MODEL
        self.fallback_models = fallback_models or config.FALLBACK_MODELS
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Ksschkw/kssrag",
            "X-Title": "KSS RAG Agent"
        }
    
    def predict(self, messages: List[Dict[str, str]]) -> str:
        """Generate a response using OpenRouter's API with fallbacks"""
        logger.info(f"Attempting to generate response with {len(messages)} messages")
        
        for model in [self.model] + self.fallback_models:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 1,
                "stop": None,
                "stream": False
            }
            
            try:
                logger.info(f"Trying model: {model}")
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=15
                )
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Parse JSON response
                response_data = response.json()
                
                # Validate response structure
                if ("choices" not in response_data or 
                    len(response_data["choices"]) == 0 or
                    "message" not in response_data["choices"][0] or
                    "content" not in response_data["choices"][0]["message"]):
                    
                    logger.warning(f"Invalid response format from {model}: {response_data}")
                    continue
                
                content = response_data["choices"][0]["message"]["content"]
                logger.info(f"Successfully used model: {model}")
                return content
                
            except requests.exceptions.Timeout:
                logger.warning(f"Model {model} timed out")
                continue
            except requests.exceptions.RequestException as e:
                logger.warning(f"Request error with model {model}: {str(e)}")
                if hasattr(e, 'response') and e.response is not None:
                    try:
                        error_data = e.response.json()
                        logger.warning(f"Error response: {error_data}")
                    except:
                        logger.warning(f"Error response text: {e.response.text}")
                continue
            except Exception as e:
                logger.warning(f"Unexpected error with model {model}: {str(e)}")
                continue
        
        # If all models fail, return a friendly error message
        error_msg = "I'm having trouble connecting to the knowledge service right now. Please try again in a moment."
        logger.error("All model fallbacks failed to respond")
        return error_msg