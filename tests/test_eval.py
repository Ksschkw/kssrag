import math

import pytest

from kssrag.eval import (
    precision_at_k,
    recall_at_k,
    hit_at_k,
    mrr_at_k,
    ndcg_at_k,
    evaluate_retriever,
    compare_retrievers,
)


RET = ["a", "b", "c", "d", "e"]
REL = {"b", "d", "x"}  # 'x' is never retrieved


def test_precision_at_k():
    assert precision_at_k(RET, REL, 3) == pytest.approx(1 / 3)
    assert precision_at_k(RET, REL, 5) == pytest.approx(2 / 5)
    assert precision_at_k(RET, REL, 0) == 0.0


def test_recall_at_k():
    assert recall_at_k(RET, REL, 3) == pytest.approx(1 / 3)
    assert recall_at_k(RET, REL, 5) == pytest.approx(2 / 3)
    assert recall_at_k(RET, set(), 5) == 0.0


def test_hit_at_k():
    assert hit_at_k(RET, REL, 1) == 0.0
    assert hit_at_k(RET, REL, 2) == 1.0


def test_mrr_at_k():
    assert mrr_at_k(RET, REL, 5) == pytest.approx(1 / 2)  # first relevant at rank 2
    assert mrr_at_k(RET, REL, 1) == 0.0


def test_ndcg_at_k():
    # relevant at ranks 2 and 4
    dcg = 1 / math.log2(3) + 1 / math.log2(5)
    idcg = 1 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)  # 3 relevant docs
    assert ndcg_at_k(RET, REL, 5) == pytest.approx(dcg / idcg)
    # perfect ranking
    assert ndcg_at_k(["b", "d"], {"b", "d"}, 2) == pytest.approx(1.0)


class _FakeStore:
    """Returns a fixed ranking regardless of query, for deterministic tests."""

    def __init__(self, ranking):
        self._ranking = ranking

    def retrieve(self, query, top_k):
        return [{"content": doc_id, "metadata": {"id": doc_id}} for doc_id in self._ranking[:top_k]]


def test_evaluate_retriever_perfect():
    store = _FakeStore(["a", "b", "c"])
    dataset = [("q1", {"a"}), ("q2", {"a", "b"})]
    scores = evaluate_retriever(store, dataset, k=3, metrics=["recall", "mrr"])
    assert scores["mrr"] == pytest.approx(1.0)   # 'a' first in both
    assert scores["recall"] == pytest.approx((1.0 + 1.0) / 2)


def test_evaluate_retriever_empty_dataset():
    store = _FakeStore(["a"])
    assert evaluate_retriever(store, [], k=3) == {
        name: 0.0 for name in ["precision", "recall", "hit", "mrr", "ndcg"]
    }


def test_unknown_metric_raises():
    store = _FakeStore(["a"])
    with pytest.raises(ValueError):
        evaluate_retriever(store, [("q", {"a"})], metrics=["bogus"])


def test_compare_retrievers():
    good = _FakeStore(["a", "b", "c"])
    bad = _FakeStore(["z", "y", "a"])
    dataset = [("q", {"a"})]
    results = compare_retrievers({"good": good, "bad": bad}, dataset, k=3, metrics=["mrr"])
    assert results["good"]["mrr"] > results["bad"]["mrr"]
