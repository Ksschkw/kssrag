from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid
import json

from kssrag.models.openrouter import OpenRouterLLM

from .core.agents import RAGAgent
from .utils.helpers import logger
from .config import config

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class StreamResponse(BaseModel):
    chunk: str
    done: bool = False

class ServerConfig(BaseModel):
    """Configuration for the FastAPI server"""
    host: str = config.SERVER_HOST
    port: int = config.SERVER_PORT
    cors_origins: List[str] = config.CORS_ORIGINS
    cors_allow_credentials: bool = config.CORS_ALLOW_CREDENTIALS
    cors_allow_methods: List[str] = config.CORS_ALLOW_METHODS
    cors_allow_headers: List[str] = config.CORS_ALLOW_HEADERS
    title: str = "KSSSwagger"
    description: str = "[kssrag](https://github.com/Ksschkw/kssrag)"
    version: str = "0.2.0"

def create_app(rag_agent: RAGAgent, server_config: Optional[ServerConfig] = None):
    """Create a FastAPI app for the RAG agent with configurable CORS"""
    if server_config is None:
        server_config = ServerConfig()
    
    app = FastAPI(
        title=server_config.title,
        description=server_config.description,
        version=server_config.version
    )
    
    # Configure CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=server_config.cors_origins,
        allow_credentials=server_config.cors_allow_credentials,
        allow_methods=server_config.cors_allow_methods,
        allow_headers=server_config.cors_allow_headers,
    )
    
    # Session management
    sessions = {}
    
    @app.post("/query")
    async def query_endpoint(request: QueryRequest):
        """Handle user queries"""
        query = request.query
        session_id = request.session_id or str(uuid.uuid4())
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        try:
            # Get or create session
            if session_id not in sessions:
                logger.info(f"Creating new session: {session_id}")
                # Create a new agent for this session
                sessions[session_id] = RAGAgent(
                    retriever=rag_agent.retriever,
                    llm=rag_agent.llm,
                    system_prompt=rag_agent.system_prompt
                )
            
            agent = sessions[session_id]
            response = agent.query(query)
            
            return {
                "query": query,
                "response": response,
                "session_id": session_id
            }
            
        except Exception as e:
            logger.error(f"Error handling query: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")
    
    @app.post("/stream")
    async def stream_query(request: QueryRequest):
        """Streaming query endpoint with Server-Sent Events"""
        query = request.query
        session_id = request.session_id or str(uuid.uuid4())
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        try:
            # Get or create session
            if session_id not in sessions:
                logger.info(f"Creating new streaming session: {session_id}")
                sessions[session_id] = RAGAgent(
                    retriever=rag_agent.retriever,
                    llm=rag_agent.llm,
                    system_prompt=rag_agent.system_prompt
                )
            
            agent = sessions[session_id]
            
            # async def generate():
            #     full_response = ""
            #     try:
            #         # Use agent's query_stream which handles context and summarization
            #         for chunk in agent.query_stream(query, top_k=5):
            #             full_response += chunk
            #             yield f"data: {json.dumps({'chunk': chunk, 'done': False})}\n\n"
                    
            #         yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"
                    
            #     except Exception as e:
            #         logger.error(f"Streaming error: {str(e)}")
            #         yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"

            async def generate():
                try:
                    # Stream tokens ONLY
                    for token in agent.query_stream(query, top_k=5):
                        if not token:
                            continue

                        yield f"data: {json.dumps({'chunk': token, 'done': False})}\n\n"

                    # Signal completion (no payload)
                    yield f"data: {json.dumps({'chunk': '', 'done': True})}\n\n"

                except Exception as e:
                    logger.error(f"Streaming error: {str(e)}")
                    yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


            
            return StreamingResponse(
                generate(), 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
            
        except Exception as e:
            logger.error(f"Streaming query failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {
            "status": "healthy", 
            "message": "KSS RAG API is running",
            "version": server_config.version
        }
    
    @app.get("/config")
    async def get_config():
        """Get current server configuration"""
        return server_config.dict()
    
    @app.get("/sessions/{session_id}/clear")
    async def clear_session(session_id: str):
        """Clear a session's conversation history"""
        if session_id in sessions:
            sessions[session_id].clear_conversation()
            return {"message": f"Session {session_id} cleared"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    
    @app.get("/")
    async def root():
        """Root endpoint with API information"""
        return {
            "message": "Welcome to KSSRAG API",
            "version": server_config.version,
            "docs": "/docs",
            "health": "/health"
        }
    
    return app, server_config