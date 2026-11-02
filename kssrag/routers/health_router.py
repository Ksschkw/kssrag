"""
Health, configuration, and root router.
Conforms to house-standard.md §2 and §11.
"""
from fastapi import APIRouter, Request
from ..schemas.health_schema import HealthResponseSchema

health_router = APIRouter()


@health_router.get("/health", response_model=HealthResponseSchema)
async def health_endpoint():
    """Liveness probe. Excludes internal versions or configs per house-standard.md §11."""
    return HealthResponseSchema(status="healthy", message="KSS RAG API is running")


@health_router.get("/config")
async def config_endpoint(request: Request):
    """Configuration retrieval endpoint."""
    server_config = getattr(request.app.state, "server_config", None)
    if server_config is not None:
        return server_config.model_dump()
    return {}


@health_router.get("/")
async def root_endpoint(request: Request):
    """Root info endpoint."""
    server_config = getattr(request.app.state, "server_config", None)
    version = getattr(server_config, "version", "0.4.0") if server_config else "0.4.0"
    return {
        "message": "Welcome to KSSRAG API",
        "version": version,
        "docs": "/docs",
        "health": "/health",
    }
