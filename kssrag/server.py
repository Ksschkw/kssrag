from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import uuid

from .core.agents import RAGAgent
from .utils.helpers import logger
from .config import config

class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class ServerConfig(BaseModel):
    """Configuration for the FastAPI server"""
    host: str = config.SERVER_HOST
    port: int = config.SERVER_PORT
    cors_origins: List[str] = config.CORS_ORIGINS
    cors_allow_credentials: bool = config.CORS_ALLOW_CREDENTIALS
    cors_allow_methods: List[str] = config.CORS_ALLOW_METHODS
    cors_allow_headers: List[str] = config.CORS_ALLOW_HEADERS
    title: str = "KSS RAG API"
    description: str = "A Retrieval-Augmented Generation API by Ksschkw"
    version: str = "0.1.0"

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
            "message": "Welcome to KSS RAG API",
            "version": server_config.version,
            "docs": "/docs",
            "health": "/health"
        }
    
    return app, server_config