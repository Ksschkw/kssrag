"""
Rank-based fusion for combining results from multiple retrievers.

Reciprocal Rank Fusion (RRF) is the method used by production search engines
(Elasticsearch, Vespa) to combine rankings from scorers whose raw scores are not
comparable — e.g. a lexical BM25 score and a semantic cosine similarity. Because
RRF operates on *rank position* rather than raw score, it needs no score
normalization and is robust to outliers.

    score(doc) = sum over retrievers of  1 / (k + rank(doc))

where rank is 1-based and k is a smoothing constant (60 is the value from the
original Cormack et al. 2009 paper and the de-facto default).
"""
from typing import Any, Callable, Dict, List, Optional

DEFAULT_RRF_K = 60


def _default_key(doc: Dict[str, Any]) -> int:
    """Identity key for a document: content plus its metadata."""
    return hash(doc["content"] + str(doc.get("metadata", "")))


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    top_k: int,
    k: int = DEFAULT_RRF_K,
    weights: Optional[List[float]] = None,
    key_fn: Callable[[Dict[str, Any]], Any] = _default_key,
) -> List[Dict[str, Any]]:
    """
    Fuse several ranked document lists into one ranking via RRF.

    Args:
        ranked_lists: one list per retriever, each already ordered best-first.
        top_k: number of documents to return.
        k: RRF smoothing constant (higher => flatter contribution from rank).
        weights: optional per-retriever weights (defaults to 1.0 each).
        key_fn: maps a document to a hashable identity for deduplication.

    Returns:
        Up to top_k documents ordered by descending fused score. Ties are broken
        deterministically by first appearance across the input lists.
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)
    elif len(weights) != len(ranked_lists):
        raise ValueError("weights must have one entry per ranked list")

    scores: Dict[Any, float] = {}
    doc_by_key: Dict[Any, Dict[str, Any]] = {}
    first_seen: Dict[Any, int] = {}
    order = 0

    for weight, ranked in zip(weights, ranked_lists):
        for rank, doc in enumerate(ranked, start=1):
            key = key_fn(doc)
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (k + rank))
            if key not in doc_by_key:
                doc_by_key[key] = doc
                first_seen[key] = order
                order += 1

    ranked_keys = sorted(
        scores.keys(),
        key=lambda key: (-scores[key], first_seen[key]),
    )
    return [doc_by_key[key] for key in ranked_keys[:top_k]]
