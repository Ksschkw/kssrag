from typing import List, Dict, Any, Optional
from ..utils.helpers import logger

class RAGAgent:
    """RAG agent implementation"""
    
    def __init__(self, retriever, llm, system_prompt: Optional[str] = None, 
                 conversation_history: Optional[List[Dict[str, str]]] = None):
        self.retriever = retriever
        self.llm = llm
        self.conversation = conversation_history or []
        self.system_prompt = system_prompt or """You are a helpful AI assistant. Use the following context to answer the user's question. 
        If you don't know the answer based on the context, say so."""
        
        # Initialize with system message if not already present
        if not any(msg.get("role") == "system" for msg in self.conversation):
            self.add_message("system", self.system_prompt)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation.append({"role": role, "content": content})
        
        # Keep conversation manageable (last 10 messages)
        if len(self.conversation) > 10:
            # Always keep the system message
            system_msg = next((msg for msg in self.conversation if msg["role"] == "system"), None)
            other_msgs = [msg for msg in self.conversation if msg["role"] != "system"]
            
            # Keep the most recent messages
            self.conversation = [system_msg] + other_msgs[-9:] if system_msg else other_msgs[-10:]
    
    def query(self, question: str, top_k: int = 5, include_context: bool = True) -> str:
        """Process a query and return a response"""
        try:
            # Retrieve relevant context
            context_docs = self.retriever.retrieve(question, top_k)
            
            if not context_docs and include_context:
                logger.warning(f"No context found for query: {question}")
                return "I couldn't find relevant information to answer your question."
            
            # Format context
            context = ""
            if include_context and context_docs:
                context = "Relevant information:\n"
                for i, doc in enumerate(context_docs, 1):
                    context += f"\n--- Document {i} ---\n{doc['content']}\n"
            
            # Add user query with context
            user_message = f"{context}\n\nQuestion: {question}" if context else question
            self.add_message("user", user_message)
            
            # Generate response
            response = self.llm.predict(self.conversation)
            
            # Add assistant response to conversation
            self.add_message("assistant", response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return "I encountered an issue processing your query. Please try again."
    
    def clear_conversation(self):
        """Clear conversation history except system message"""
        system_msg = next((msg for msg in self.conversation if msg["role"] == "system"), None)
        self.conversation = [system_msg] if system_msg else []