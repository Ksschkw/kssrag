"""
Retrieval quality metrics.

All functions take:
  retrieved: the ranked list of document ids returned by a retriever (best first)
  relevant:  the set of document ids that are truly relevant for the query

and evaluate the ranking at cutoff k. Ids may be any hashable value as long as
the retriever and the ground-truth labels use the same id space.

Metrics implemented:
  - precision@k : fraction of the top-k that are relevant
  - recall@k    : fraction of all relevant docs found in the top-k
  - mrr@k       : reciprocal rank of the first relevant doc (0 if none in top-k)
  - ndcg@k      : normalized discounted cumulative gain (binary relevance)
  - hit@k       : 1.0 if any relevant doc appears in the top-k else 0.0
"""
import math
from typing import Hashable, List, Sequence, Set


def precision_at_k(retrieved: Sequence[Hashable], relevant: Set[Hashable], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / len(top)


def recall_at_k(retrieved: Sequence[Hashable], relevant: Set[Hashable], k: int) -> float:
    if not relevant:
        return 0.0
    top = retrieved[:k]
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / len(relevant)


def hit_at_k(retrieved: Sequence[Hashable], relevant: Set[Hashable], k: int) -> float:
    return 1.0 if any(doc_id in relevant for doc_id in retrieved[:k]) else 0.0


def mrr_at_k(retrieved: Sequence[Hashable], relevant: Set[Hashable], k: int) -> float:
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved: Sequence[Hashable], relevant: Set[Hashable], k: int) -> float:
    """Discounted cumulative gain with binary relevance (gain 1 if relevant)."""
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved: Sequence[Hashable], relevant: Set[Hashable], k: int) -> float:
    """DCG normalized by the ideal DCG (all relevant docs ranked first)."""
    if not relevant:
        return 0.0
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / idcg


# Registry so the runner can iterate metrics by name.
METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "hit": hit_at_k,
    "mrr": mrr_at_k,
    "ndcg": ndcg_at_k,
}
