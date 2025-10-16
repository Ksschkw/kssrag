import requests
import json
from typing import List, Dict, Any, Optional, Generator
from ..utils.helpers import logger
from ..config import config

class OpenRouterLLM:
    """OpenRouter LLM interface with streaming support"""
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, 
                 fallback_models: Optional[List[str]] = None, stream: bool = False):
        self.api_key = api_key or config.OPENROUTER_API_KEY
        self.model = model or config.DEFAULT_MODEL
        self.fallback_models = fallback_models or config.FALLBACK_MODELS
        self.stream = stream
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/Ksschkw/kssrag",
            "X-Title": "KSS RAG Agent"
        }
    
    def predict(self, messages: List[Dict[str, str]]) -> str:
        """Generate response with fallback models"""
        if self.stream:
            full_response = ""
            for chunk in self.predict_stream(messages):
                full_response += chunk
            return full_response
        
        logger.info(f"Generating response with {len(messages)} messages")
        
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
                logger.info(f"Using model: {model}")
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                if ("choices" not in response_data or 
                    len(response_data["choices"]) == 0 or
                    "message" not in response_data["choices"][0] or
                    "content" not in response_data["choices"][0]["message"]):
                    
                    logger.warning(f"Invalid response format from {model}: {response_data}")
                    continue
                
                content = response_data["choices"][0]["message"]["content"]
                logger.info(f"Successfully generated response with model: {model}")
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
        
        error_msg = "Unable to generate response from available models. Please try again."
        logger.error("All model fallbacks failed to respond")
        return error_msg
    
    def predict_stream(self, messages: List[Dict[str, str]]) -> Generator[str, None, None]:
        """Stream response from OpenRouter API"""
        logger.info(f"Streaming response with {len(messages)} messages")
        
        for model in [self.model] + self.fallback_models:
            payload = {
                "model": model,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 1,
                "stop": None,
                "stream": True
            }
            
            try:
                logger.info(f"Streaming with model: {model}")
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                    stream=True
                )
                
                response.raise_for_status()
                
                for line in response.iter_lines():
                    if line:
                        line = line.decode('utf-8')
                        if line.startswith('data: '):
                            data = line[6:]
                            if data.strip() == '[DONE]':
                                logger.info("Stream completed successfully")
                                return
                            try:
                                chunk_data = json.loads(data)
                                if ('choices' in chunk_data and 
                                    len(chunk_data['choices']) > 0 and
                                    'delta' in chunk_data['choices'][0] and
                                    'content' in chunk_data['choices'][0]['delta']):
                                    
                                    content = chunk_data['choices'][0]['delta']['content']
                                    if content:
                                        yield content
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse stream chunk: {str(e)}")
                                continue
                
                logger.info(f"Successfully streamed from model: {model}")
                return
                
            except Exception as e:
                logger.warning(f"Streaming failed with model {model}: {str(e)}")
                continue
        
        error_msg = "Unable to stream response from available models. Please try again."
        logger.error("All model fallbacks failed for streaming")
        yield error_msg