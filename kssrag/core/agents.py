from typing import Generator, List, Dict, Any, Optional
from ..utils.helpers import logger

class RAGAgent:
    """RAG agent implementation with discrete conversation summaries"""
    
    def __init__(self, retriever, llm, system_prompt: Optional[str] = None, 
                 conversation_history: Optional[List[Dict[str, str]]] = None):
        self.retriever = retriever
        self.llm = llm
        self.conversation = conversation_history or []
        self.system_prompt = system_prompt or """You are a helpful AI assistant. Use the following context to answer the user's question. 
        If you don't know the answer based on the context, say so."""
        self.conversation_summaries = []  # Discrete summaries instead of single blob

        logger.info(f"RAGAgent initialized with {len(conversation_history or [])} history messages")
        logger.info(f"System prompt: {self.system_prompt[:100]}..." if self.system_prompt else "No system prompt")

        
        # Initialize with system message if not already present
        if not any(msg.get("role") == "system" for msg in self.conversation):
            self.add_message("system", self.system_prompt)
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.conversation.append({"role": role, "content": content})
        
        # Keep conversation manageable (last 15 messages)
        if len(self.conversation) > 15:
            self._smart_trim_conversation()
    
    def _smart_trim_conversation(self):
        """Trim conversation while preserving system message and recent exchanges"""
        if len(self.conversation) <= 15:
            return
        
        original_count = len(self.conversation)
        # Always keep system message
        system_msg = next((msg for msg in self.conversation if msg["role"] == "system"), None)
        
        # Keep recent messages (last 14)
        recent_messages = self.conversation[-14:]
        
        # Rebuild: system + recent
        new_conv = []
        if system_msg:
            new_conv.append(system_msg)
        new_conv.extend(recent_messages)
        
        self.conversation = new_conv
        
        # Also trim summaries to match conversation scope
        if len(self.conversation_summaries) > 7:
            self.conversation_summaries = self.conversation_summaries[-7:]
        logger.info(f"Trimmed conversation from {original_count} to {len(self.conversation)} messages")
    
    def _build_context(self, context_docs: List[Dict[str, Any]]) -> str:
        """Build context string from documents"""
        if not context_docs:
            return ""
        
        context = "Relevant information:\n"
        for i, doc in enumerate(context_docs, 1):
            context += f"\n--- Document {i} ---\n{doc['content']}\n"
        return context
    
    def _build_messages(self, question: str, context: str = "") -> List[Dict[str, str]]:
        """Build messages for LLM including context and conversation summaries"""
        # Start with conversation history
        messages = self.conversation.copy()
        
        logger.info(f"Building messages for query: '{question}'")
        logger.info(f"Conversation history: {len(self.conversation)} messages")
        logger.info(f"Active summaries: {len(self.conversation_summaries)}")
        logger.info(f"Retrieved context: {len(context)} chars" if context else "No retrieved context")

        # Add conversation summaries as context if available
        if self.conversation_summaries:
            logger.info(f"Using summaries: {self.conversation_summaries}")
            summary_context = "Previous conversation context:\n" + "\n".join(
                f"- {summary}" for summary in self.conversation_summaries[-3:]  # Last 3 summaries
            )
            messages.append({
                "role": "system", 
                "content": summary_context
            })
        
        # Add retrieved document context
        user_message = f"{context}\n\nQuestion: {question}" if context else question
        
        # ✅ FIX: Always append new user message (don't replace existing ones)
        messages.append({"role": "user", "content": user_message})
        
        # Add stealth summarization instruction for ongoing conversations
        if len(self.conversation) >= 1:  # More than just system + current user message + 2nd Query
            summary_instruction = self._create_summary_instruction()
            messages.append({"role": "system", "content": summary_instruction})
            logger.info(f" Summary instruction added to prompt: {len(summary_instruction)} chars")
            logger.debug(f"Instruction content: {summary_instruction}")

        logger.info(f" Final message count to LLM: {len(messages)}")
        return messages
    
    def _create_summary_instruction(self) -> str:
        """Create the stealth summarization instruction with examples"""
        return """IMPORTANT: You MUST follow this response structure:

    [YOUR MAIN RESPONSE TO THE USER GOES HERE]

    [SUMMARY_START]
    Key context from this exchange: [Brief summary of new information]
    [SUMMARY_END]

    EXAMPLES:
    If user says "My name is John", your summary should be: "User's name is John"
    If user says "I prefer formal language", your summary should be: "User prefers formal communication style"
    If user shares a preference, summarize it: "User mentioned [preference]"

    RULES:
    - ALWAYS include the summary section
    - Use EXACT markers: [SUMMARY_START] and [SUMMARY_END]
    - Keep summary 1-2 sentences
    - Focus on user preferences, names, important context

    The summary will be automatically hidden from the user."""
    
    # def _extract_summary_and_response(self, full_response: str) -> tuple[str, Optional[str]]:
    #     """Extract summary from response and return clean user response - handles partial markers"""
    #     summary_start = "[SUMMARY_START]"
    #     summary_end = "[SUMMARY_END]"
        
    #     # Check if we have complete markers
    #     if summary_start in full_response and summary_end in full_response:
    #         start_idx = full_response.find(summary_start) + len(summary_start)
    #         end_idx = full_response.find(summary_end)
            
    #         summary = full_response[start_idx:end_idx].strip()
    #         user_response = full_response[:full_response.find(summary_start)].strip()
            
    #         logger.info(f"✅ SUCCESS: Summary extracted and separated from user response")
    #         logger.info(f"User response length: {len(user_response)} chars")
    #         logger.info(f"Summary extracted: '{summary}'")
    #         return user_response, summary
        
    #     # Check if we have partial markers (common in streaming)
    #     elif summary_start in full_response:
    #         # We have start marker but no end marker - extract what we can
    #         start_idx = full_response.find(summary_start) + len(summary_start)
    #         potential_summary = full_response[start_idx:].strip()
            
    #         # Clean up any partial end markers or weird formatting
    #         if potential_summary:
    #             # Remove any trailing partial markers or whitespace
    #             cleaned_summary = potential_summary.split('[SUMMARY_')[0].split('[SUMMARY')[0].strip()
    #             user_response = full_response[:full_response.find(summary_start)].strip()
                
    #             if cleaned_summary and len(cleaned_summary) > 10:  # Only if meaningful content
    #                 logger.info(f"⚠️  Partial summary extracted (missing end marker): '{cleaned_summary}'")
    #                 return user_response, cleaned_summary
            
    #         logger.info("❌ Incomplete summary markers found")
    #         return full_response, None
        
    #     logger.info("❌ No summary markers found, returning full response")
    #     logger.info(f"Full response length: {len(full_response)} chars")
    #     return full_response, None

    def _extract_summary_and_response(self, full_response: str) -> tuple[str, Optional[str]]:
        """Extract summary from response and return clean user response - handles partial markers"""
        # Keep original markers for backward compatibility
        summary_start = "[SUMMARY_START]"
        summary_end = "[SUMMARY_END]"
        
        # NEW: Normalize the response first (improvement from new version)
        normalized = full_response.replace('\n', ' ').replace('\r', ' ').strip()
        
        # Check if we have complete markers - KEEP original logic but use normalized
        if summary_start in normalized and summary_end in normalized:
            start_idx = normalized.find(summary_start) + len(summary_start)
            end_idx = normalized.find(summary_end)
            
            summary = normalized[start_idx:end_idx].strip()
            user_response = normalized[:normalized.find(summary_start)].strip()
            
            logger.info(f"✅ SUCCESS: Summary extracted and separated from user response")
            logger.info(f"User response length: {len(user_response)} chars")
            logger.info(f"Summary extracted: '{summary}'")
            
            # NEW: Add validation from improved version
            if not summary or len(summary) < 5:
                logger.info("❌ Summary too short, returning full response")
                return full_response.strip(), None
                
            return user_response, summary
        
        # Check if we have partial markers (common in streaming) - IMPROVED logic
        elif summary_start in normalized:
            # We have start marker but no end marker - extract what we can
            start_idx = normalized.find(summary_start) + len(summary_start)
            
            # NEW: Take reasonable chunk (200 chars) instead of everything
            potential_summary = normalized[start_idx:start_idx+200].strip()
            
            # COMBINED: Clean up from both versions
            if potential_summary:
                # Clean up any partial markers or weird formatting
                cleaned_summary = (potential_summary
                                .split('[SUMMARY_')[0]
                                .split('[SUMMARY')[0]
                                .split('[')[0]  # NEW from improved version
                                .split('\n')[0]  # NEW from improved version
                                .strip())
                
                user_response = normalized[:normalized.find(summary_start)].strip()
                
                # COMBINED validation: meaningful content check
                if cleaned_summary and len(cleaned_summary) >= 10:  # Original threshold
                    logger.info(f"⚠️  Partial summary extracted (missing end marker): '{cleaned_summary}'")
                    # NEW: Additional validation
                    if len(cleaned_summary) >= 5:  # Improved version threshold
                        return user_response, cleaned_summary
                
            logger.info("❌ Incomplete summary markers found")
            return full_response.strip(), None  # NEW: strip for consistency
        
        # No markers found - KEEP original but with normalization
        logger.info("❌ No summary markers found, returning full response")
        logger.info(f"Full response length: {len(full_response)} chars")
        return full_response.strip(), None  # NEW: strip for consistency
    
    def _add_conversation_summary(self, new_summary: str):
        """Add a new discrete conversation summary"""
        if not new_summary or new_summary.lower() == "none":
            logger.info("🔄 No summary to add (empty or 'none')")
            return
        
        # Add as a new discrete summary
        self.conversation_summaries.append(new_summary)
        logger.info(f"📝 ADDED Summary #{len(self.conversation_summaries)}: '{new_summary}'")

        # Keep only recent summaries (last 7)
        if len(self.conversation_summaries) > 7:
            self.conversation_summaries = self.conversation_summaries[-7:]
            removed = self.conversation_summaries.pop(0)
            logger.info(f"🗑️  DROPPED Oldest summary: '{removed}'")
            logger.info(f"📊 Summary count maintained at {len(self.conversation_summaries)}")
        logger.info(f"Added conversation summary #{len(self.conversation_summaries)}: {new_summary}")
    
    def query(self, question: str, top_k: int = 5, include_context: bool = True) -> str:
        """Process a query with stealth conversation summarization"""
        try:
            # Retrieve relevant context
            logger.info(f"🔍 QUERY START: '{question}' (top_k: {top_k})")
            context_docs = self.retriever.retrieve(question, top_k)
            logger.info(f"📄 Retrieved {len(context_docs)} context documents")
            
            if not context_docs and include_context:
                logger.warning(f"No context found for query: {question}")
                return "I couldn't find relevant information to answer your question."
            
            # Format context
            context = self._build_context(context_docs) if include_context and context_docs else ""
            
            # Build messages
            messages = self._build_messages(question, context)
            
            # Generate response
            full_response = self.llm.predict(messages)
            logger.info(f"🤖 LLM response received: {len(full_response)} chars")
            
            # Extract summary and clean response
            user_response, conversation_summary = self._extract_summary_and_response(full_response)
            
            # Add new summary if found
            if conversation_summary:
                self._add_conversation_summary(conversation_summary)
                logger.info(" Summary processing completed successfully")
            else:
                logger.info("Bitch No summary generated for this exchange")
            
            # Add assistant response to conversation (clean version only)
            self.add_message("assistant", user_response)
            
            logger.info(f"💬 Final user response: {len(user_response)} chars")
            return user_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # logger.error(f"💥 QUERY FAILED: {str(e)}")
            return "I encountered an issue processing your query. Please try again."
    
    def query_stream(self, question: str, top_k: int = 5) -> Generator[str, None, None]:
        """
        Professional-grade streaming with multiple fallback strategies
        """
        try:
            logger.info(f"🌊 STREAMING QUERY START: '{question}'")
            
            # Strategy 1: Try true streaming first
            if hasattr(self.llm, 'predict_stream'):
                try:
                    yield from self._stream_with_summary_protection(question, top_k)
                    return
                except Exception as stream_error:
                    logger.warning(f"Streaming failed, falling back: {stream_error}")
            
            # Strategy 2: Fallback to simulated streaming
            logger.info("🔄 Falling back to simulated streaming")
            yield from self._simulated_streaming(question, top_k)
            
        except Exception as e:
            logger.error(f"💥 ALL STREAMING STRATEGIES FAILED: {str(e)}")
            yield f"Error: {str(e)}"

    def _stream_with_summary_protection(self, question: str, top_k: int) -> Generator[str, None, None]:
        """True streaming with better error handling"""
        try:
            relevant_docs = self.retriever.retrieve(question, top_k=top_k)
            context = self._build_context(relevant_docs)
            messages = self._build_messages(question, context)
            
            buffer = ""
            summary_started = False
            
            for chunk in self.llm.predict_stream(messages):
                buffer += chunk
                
                # Check for summary markers
                if any(marker in chunk for marker in ['[SUMMARY', 'SUMMARY_']):
                    if not summary_started:
                        logger.info("🚨 Summary markers detected - cutting stream")
                        summary_started = True
                        clean_part = self._extract_clean_content(buffer)
                        if clean_part:
                            yield clean_part
                        # Don't break here - let the method complete naturally
                        continue
                
                if not summary_started:
                    yield chunk
            
            # Process the complete response
            self._process_complete_response(buffer)
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            raise  # Re-raise to trigger fallback

    def _process_complete_response(self, full_response: str):
        """Process complete response and extract summary"""
        user_response, conversation_summary = self._extract_summary_and_response(full_response)
        
        if conversation_summary:
            logger.info(f"📝 Summary extracted: '{conversation_summary}'")
            self._add_conversation_summary(conversation_summary)
        
        self.add_message("assistant", user_response)

    def _simulated_streaming(self, question: str, top_k: int) -> Generator[str, None, None]:
        """Simulated streaming that guarantees no summary leakage"""
        relevant_docs = self.retriever.retrieve(question, top_k=top_k)
        context = self._build_context(relevant_docs)
        messages = self._build_messages(question, context)
        
        # Get complete response
        complete_response = self.llm.predict(messages)
        
        # Extract clean response
        user_response, conversation_summary = self._extract_summary_and_response(complete_response)
        
        if conversation_summary:
            logger.info(f"📝 Summary extracted: '{conversation_summary}'")
            self._add_conversation_summary(conversation_summary)
        
        self.add_message("assistant", user_response)
        
        # Simulate streaming (smaller chunks for better UX)
        chunk_size = 2  # Even smaller chunks for smoother streaming
        for i in range(0, len(user_response), chunk_size):
            yield user_response[i:i+chunk_size]
            import time
            time.sleep(0.02)  # Slightly longer delay for readability

    def _extract_clean_content(self, buffer: str) -> str:
        """Extract clean content before any summary markers"""
        markers = ['[SUMMARY_START]', '[SUMMARY', 'SUMMARY_']
        for marker in markers:
            if marker in buffer:
                return buffer.split(marker)[0].strip()
        return buffer.strip()
    
    def clear_conversation(self):
        """Clear conversation history except system message and summaries"""
        system_msg = next((msg for msg in self.conversation if msg["role"] == "system"), None)
        self.conversation = [system_msg] if system_msg else []
        # I wanna Keep conversation summaries - they're the compressed memory!
        # self.conversation_summaries = []  TO:DO(If bug noticed) # Optional: clear summaries too
    
    def get_conversation_context(self) -> Dict[str, Any]:
        context = {
            "summary_count": len(self.conversation_summaries),
            "summaries": self.conversation_summaries,
            "message_count": len(self.conversation),
            "recent_messages": [f"{msg['role']}: {msg['content'][:50]}..." for msg in self.conversation[-3:]]
        }
        logger.info(f"📊 Context snapshot: {context}")
        return context