"""
Backward compatibility re-export facade for configuration.
Canonical configuration and feature flags live in kssrag.core.config.
"""
from .core.config import (
    Config,
    VectorStoreType,
    ChunkerType,
    RetrieverType,
    config,
    log_feature_flags,
)

__all__ = [
    "Config",
    "VectorStoreType",
    "ChunkerType",
    "RetrieverType",
    "config",
    "log_feature_flags",
]
