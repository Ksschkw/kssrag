from typing import List, Dict, Any, Optional
from ..utils.helpers import logger

class BaseRetriever:
    """Base class for retrievers"""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve documents based on query"""
        raise NotImplementedError("Subclasses must implement this method")

class SimpleRetriever(BaseRetriever):
    """Simple retriever using only vector store"""
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        return self.vector_store.retrieve(query, top_k)

class HybridRetriever(BaseRetriever):
    """Hybrid retriever with fuzzy matching for specific entities"""
    
    def __init__(self, vector_store, entity_names: Optional[List[str]] = None):
        super().__init__(vector_store)
        self.entity_names = entity_names or []
    
    def extract_entities(self, query: str) -> List[str]:
        """Extract entities from query using fuzzy matching"""
        from rapidfuzz import process, fuzz
        
        extracted_entities = []
        query_lower = query.lower()
        
        # Check for exact matches first
        for entity in self.entity_names:
            if entity in query_lower:
                extracted_entities.append(entity)
        
        # Use fuzzy matching for partial matches
        if not extracted_entities and self.entity_names:
            matches = process.extract(query, self.entity_names, scorer=fuzz.partial_ratio, limit=5)
            extracted_entities = [match[0] for match in matches if match[1] > 80]
        
        return extracted_entities
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        # First get results from vector store
        results = self.vector_store.retrieve(query, top_k * 2)
        
        # If we have entity names, boost documents that mention extracted entities
        if self.entity_names:
            extracted_entities = self.extract_entities(query)
            
            if extracted_entities:
                # Boost scores for documents containing extracted entities
                scored_results = []
                
                for doc in results:
                    score = 1.0
                    content_lower = doc["content"].lower()
                    
                    # Boost score if any extracted entity is mentioned
                    for entity in extracted_entities:
                        if entity in content_lower:
                            score += 0.5
                            break
                    
                    scored_results.append((doc, score))
                
                # Sort by score and return top_k
                scored_results.sort(key=lambda x: x[1], reverse=True)
                return [doc for doc, _ in scored_results[:top_k]]
        
        return results[:top_k]