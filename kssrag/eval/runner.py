"""
Retrieval evaluation runner.

Scores a vector store (or any object with a ``retrieve(query, top_k)`` method)
against a labeled dataset of queries and their relevant document ids, returning
mean metrics across all queries.

A labeled dataset is a list of (query, relevant_ids) pairs::

    dataset = [
        ("how do I reset my password", {"doc_3", "doc_7"}),
        ("what are the office hours",   {"doc_1"}),
    ]

Documents must expose a stable id. By default the runner reads it from
``doc["metadata"]["id"]``; pass a custom ``id_fn`` if your ids live elsewhere.
"""
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Set, Tuple

from .metrics import METRICS

LabeledDataset = Sequence[Tuple[str, Set[Hashable]]]


def _default_id_fn(doc: Dict) -> Hashable:
    return doc["metadata"]["id"]


def evaluate_retriever(
    retriever,
    dataset: LabeledDataset,
    k: int = 10,
    metrics: Optional[Sequence[str]] = None,
    id_fn: Callable[[Dict], Hashable] = _default_id_fn,
) -> Dict[str, float]:
    """
    Return the mean of each metric over all queries in the dataset.

    Args:
        retriever: object with ``retrieve(query, top_k) -> list[doc]``.
        dataset: list of (query, relevant_ids).
        k: evaluation cutoff (also the number of docs retrieved per query).
        metrics: metric names to compute (defaults to all in METRICS).
        id_fn: maps a returned document to its id.

    Returns:
        {metric_name: mean_score}. Empty dataset yields 0.0 for every metric.
    """
    metric_names = list(metrics) if metrics is not None else list(METRICS)
    for name in metric_names:
        if name not in METRICS:
            raise ValueError(f"Unknown metric '{name}'. Available: {list(METRICS)}")

    totals = {name: 0.0 for name in metric_names}
    n = 0
    for query, relevant in dataset:
        retrieved_docs = retriever.retrieve(query, k)
        retrieved_ids = [id_fn(doc) for doc in retrieved_docs]
        for name in metric_names:
            totals[name] += METRICS[name](retrieved_ids, set(relevant), k)
        n += 1

    if n == 0:
        return {name: 0.0 for name in metric_names}
    return {name: totals[name] / n for name in metric_names}


def compare_retrievers(
    named_retrievers: Dict[str, object],
    dataset: LabeledDataset,
    k: int = 10,
    metrics: Optional[Sequence[str]] = None,
    id_fn: Callable[[Dict], Hashable] = _default_id_fn,
) -> Dict[str, Dict[str, float]]:
    """Evaluate several named retrievers on the same dataset. Returns name -> metrics."""
    return {
        name: evaluate_retriever(retriever, dataset, k=k, metrics=metrics, id_fn=id_fn)
        for name, retriever in named_retrievers.items()
    }


def format_comparison(results: Dict[str, Dict[str, float]], k: int = 10) -> str:
    """Render compare_retrievers output as a fixed-width table for the console."""
    if not results:
        return "(no results)"
    metric_names = list(next(iter(results.values())).keys())
    name_w = max(len("retriever"), *(len(n) for n in results))
    header = "retriever".ljust(name_w) + "  " + "  ".join(f"{m}@{k}".rjust(10) for m in metric_names)
    lines = [header, "-" * len(header)]
    for name, scores in results.items():
        row = name.ljust(name_w) + "  " + "  ".join(f"{scores[m]:.4f}".rjust(10) for m in metric_names)
        lines.append(row)
    return "\n".join(lines)
