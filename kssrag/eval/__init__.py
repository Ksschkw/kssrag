"""Retrieval evaluation harness for KSS RAG.

Measure retrieval quality (recall@k, precision@k, MRR, nDCG) so you can pick the
vector store and chunker that actually work best on your data instead of guessing.
"""
from .metrics import (
    precision_at_k,
    recall_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    dcg_at_k,
    METRICS,
)
from .runner import (
    evaluate_retriever,
    compare_retrievers,
    format_comparison,
)

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "hit_at_k",
    "mrr_at_k",
    "ndcg_at_k",
    "dcg_at_k",
    "METRICS",
    "evaluate_retriever",
    "compare_retrievers",
    "format_comparison",
]
