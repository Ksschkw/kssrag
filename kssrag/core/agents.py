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
    
    # def add_message(self, role: str, content: str):
    #     """Add a message to the conversation history"""
    #     self.conversation.append({"role": role, "content": content})
        
    #     # Keep conversation manageable (last 15 messages)
    #     if len(self.conversation) > 15:
    #         self._smart_trim_conversation()

    def add_message(self, role: str, content: str):
        """Add a message to the conversation history (with simple dedupe for assistant)."""
        content = content.strip()
        # Prevent adding empty messages
        if not content:
            logger.info("Attempted to add empty message – ignored.")
            return

        # If last message is identical assistant content, skip to avoid duplicates
        if self.conversation:
            last = self.conversation[-1]
            if role == "assistant" and last.get("role") == "assistant":
                if last.get("content", "").strip() == content:
                    logger.info("Duplicate assistant message suppressed.")
                    return

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
        """
        Build messages for the LLM including context, conversation history, and summaries.

        Improvements:
        - Prevents token explosion by trimming conversation smartly
        - Injects last 3 summaries only
        - Adds stealth summarization only if there are at least 2 user-assistant exchanges
        - Preserves system messages and formatting
        """
        # Start with system + conversation history
        messages: List[Dict[str, str]] = []

        # Always include system message at top
        system_msg = next((msg for msg in self.conversation if msg["role"] == "system"), None)
        if system_msg:
            messages.append(system_msg)
        
        # Keep only last 12 user/assistant messages to prevent token overload
        conversation_tail = [msg for msg in self.conversation if msg["role"] != "system"][-12:]
        messages.extend(conversation_tail)

        logger.info(f"Building messages for query: '{question}'")
        logger.info(f"Conversation tail: {len(conversation_tail)} messages")
        logger.info(f"Active summaries: {len(self.conversation_summaries)}")
        logger.info(f"Context length: {len(context)} chars" if context else "No retrieved context")

        # Inject last 5 summaries safely as a system message
        if self.conversation_summaries:
            summaries_to_use = self.conversation_summaries[-5:]
            summary_context = "Previous conversation context:\n" + "\n".join(f"- {s}" for s in summaries_to_use)
            messages.append({"role": "system", "content": summary_context})
            logger.info(f"Injected {len(summaries_to_use)} conversation summaries")

        # Add the user's current question + retrieved context
        user_content = f"{context}\n\nQuestion: {question}" if context else question
        messages.append({"role": "user", "content": user_content})

        # Add stealth summarization only if conversation has at least 2 user-assistant pairs
        exchange_count = sum(1 for msg in self.conversation if msg["role"] != "system") // 2
        if exchange_count >= 2:
            summary_instruction = self._create_summary_instruction()
            messages.append({"role": "system", "content": summary_instruction})
            logger.info(f"Stealth summary instruction added ({len(summary_instruction)} chars)")

        logger.info(f"Final message count to LLM: {len(messages)}")
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

    def _extract_summary_and_response(self, full_response: str) -> tuple[str, Optional[str]]:
        """Extract summary from response and return clean user response safely."""
        if not full_response:
            return "", None

        summary_start = "[SUMMARY_START]"
        summary_end = "[SUMMARY_END]"

        original = full_response
        normalized = original.replace('\r\n', '\n').replace('\r', '\n')

        # Case 1: Full summary markers
        if summary_start in normalized and summary_end in normalized:
            start_idx = normalized.find(summary_start) + len(summary_start)
            end_idx = normalized.find(summary_end)
            summary = normalized[start_idx:end_idx].strip()

            user_response = original.split(summary_start)[0].strip()

            if not summary or len(summary) < 5:
                logger.info("Summary too short or invalid – returning full response as user response")
                return original.strip(), None

            return user_response, summary

        # Case 2: Partial summary start only
        if summary_start in normalized:
            start_idx = normalized.find(summary_start) + len(summary_start)
            potential = normalized[start_idx:start_idx + 200].strip()

            cleaned_summary = (
                potential
                .split('[SUMMARY_')[0]
                .split('[SUMMARY')[0]
                .split('[')[0]
                .strip()
            )

            user_response = original.split(summary_start)[0].strip()

            if cleaned_summary and len(cleaned_summary) >= 10:
                logger.info("Partial summary extracted safely")
                return user_response, cleaned_summary

            logger.info("Partial summary invalid or too short")
            return original.strip(), None

        # Case 3: No markers
        return original.strip(), None
    
    def _add_conversation_summary(self, new_summary: str):
        """Add a new discrete conversation summary"""
        if not new_summary or new_summary.lower() == "none":
            logger.info(" No summary to add (empty or 'none')")
            return

        new_summary = new_summary.strip()
        if not new_summary:
            logger.info(" No summary to add after strip")
            return

        # Append new summary
        self.conversation_summaries.append(new_summary)
        logger.info(f" ADDED Summary #{len(self.conversation_summaries)}: '{new_summary}'")

        # Keep only recent summaries (last 7)
        if len(self.conversation_summaries) > 7:
            self.conversation_summaries = self.conversation_summaries[-7:]
            logger.info(f" Summary count trimmed to {len(self.conversation_summaries)}")

    
    def query(self, question: str, top_k: int = 5, include_context: bool = True) -> str:
        """Process a query with stealth conversation summarization"""
        try:
            # Retrieve relevant context
            logger.info(f" QUERY START: '{question}' (top_k: {top_k})")
            context_docs = self.retriever.retrieve(question, top_k)
            logger.info(f" Retrieved {len(context_docs)} context documents")
            
            if not context_docs and include_context:
                logger.warning(f"No context found for query: {question}")
                return "I couldn't find relevant information to answer your question."
            
            # Format context
            context = self._build_context(context_docs) if include_context and context_docs else ""
            
            # Build messages
            messages = self._build_messages(question, context)
            
            # Generate response
            full_response = self.llm.predict(messages)
            logger.info(f" LLM response received: {len(full_response)} chars")
            
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
            
            logger.info(f" Final user response: {len(user_response)} chars")
            return user_response
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            # logger.error(f" QUERY FAILED: {str(e)}")
            return "I encountered an issue processing your query. Please try again."
    
    def query_stream(self, question: str, top_k: int = 5) -> Generator[str, None, None]:
        """
        Professional-grade streaming with multiple fallback strategies
        """
        try:
            logger.info(f" STREAMING QUERY START: '{question}'")
            
            # Strategy 1: Try true streaming first
            if hasattr(self.llm, 'predict_stream'):
                try:
                    yield from self._stream_with_summary_protection(question, top_k)
                    return
                except Exception as stream_error:
                    logger.warning(f"Streaming failed, falling back: {stream_error}")
            
            # Strategy 2: Fallback to simulated streaming
            logger.info(" Falling back to simulated streaming")
            yield from self._simulated_streaming(question, top_k)
            
        except Exception as e:
            logger.error(f" ALL STREAMING STRATEGIES FAILED: {str(e)}")
            yield f"Error: {str(e)}"

    # def _stream_with_summary_protection(self, question: str, top_k: int) -> Generator[str, None, None]:
    #     """Streaming-safe: never leak summary markers mid-stream."""
    #     relevant_docs = self.retriever.retrieve(question, top_k=top_k)
    #     context = self._build_context(relevant_docs)
    #     messages = self._build_messages(question, context)

    #     buffer = ""
    #     summary_buffer = ""
    #     in_summary = False

    #     for chunk in self.llm.predict_stream(messages):
    #         buffer += chunk

    #         # Detect summary start
    #         if '[SUMMARY_START]' in buffer:
    #             in_summary = True
    #             clean_part = buffer.split('[SUMMARY_START]')[0].strip()
    #             if clean_part:
    #                 yield clean_part
    #             summary_buffer = buffer.split('[SUMMARY_START]')[1]
    #             buffer = ""
    #             continue

    #         if in_summary:
    #             summary_buffer += chunk
    #             if '[SUMMARY_END]' in summary_buffer:
    #                 in_summary = False
    #                 summary_content = summary_buffer.split('[SUMMARY_END]')[0].strip()
    #                 if summary_content:
    #                     self._add_conversation_summary(summary_content)
    #                     logger.info(f"Summary extracted in stream: '{summary_content}'")
    #                 buffer = summary_buffer.split('[SUMMARY_END]')[1]  # remainder
    #                 summary_buffer = ""
    #                 if buffer:
    #                     yield buffer.strip()
    #                 buffer = ""
    #             continue

    #         if not in_summary:
    #             yield chunk

    #     # Flush leftover buffer
    #     if buffer.strip() and not in_summary:
    #         yield buffer.strip()
    #     elif in_summary:
    #         logger.info("Leftover buffer contains partial summary – discarded to prevent marker leak")

    def _stream_with_summary_protection(self, question: str, top_k: int) -> Generator[str, None, None]:
        """Token-only streaming. Never reconstruct or re-emit content."""
        relevant_docs = self.retriever.retrieve(question, top_k=top_k)
        context = self._build_context(relevant_docs)
        messages = self._build_messages(question, context)

        buffer = ""

        for chunk in self.llm.predict_stream(messages):
            buffer += chunk

            # The moment summary markers appear, stop streaming to client
            if '[SUMMARY_START]' in buffer or 'SUMMARY_' in buffer:
                logger.info("Summary marker detected — stopping client stream")
                break

            # Yield ONLY raw tokens
            yield chunk

        # After streaming finishes, process full response exactly once
        self._process_complete_response(buffer)

    def _process_complete_response(self, full_response: str):
        """Process complete response and extract summary"""
        user_response, conversation_summary = self._extract_summary_and_response(full_response)

        if conversation_summary:
            logger.info(f" Summary extracted: '{conversation_summary}'")
            self._add_conversation_summary(conversation_summary)

        # extra guard: only add assistant message if different from last assistant message
        if user_response:
            last = self.conversation[-1] if self.conversation else None
            if not (last and last.get("role") == "assistant" and last.get("content", "").strip() == user_response.strip()):
                self.add_message("assistant", user_response)
            else:
                logger.info("Skipped adding duplicate assistant message in _process_complete_response.")

    def _simulated_streaming(self, question: str, top_k: int) -> Generator[str, None, None]:
        """Simulated streaming that guarantees no summary leakage."""
        relevant_docs = self.retriever.retrieve(question, top_k=top_k)
        context = self._build_context(relevant_docs)
        messages = self._build_messages(question, context)

        complete_response = self.llm.predict(messages)
        user_response, conversation_summary = self._extract_summary_and_response(complete_response)

        if conversation_summary:
            self._add_conversation_summary(conversation_summary)

        self.add_message("assistant", user_response)

        # Simulate streaming chunks
        chunk_size = 2
        for i in range(0, len(user_response), chunk_size):
            yield user_response[i:i + chunk_size]
            import time
            time.sleep(0.02)


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
        logger.info(f" Context snapshot: {context}")
        return context