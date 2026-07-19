"""
Retrieval evaluation example.

Measures retrieval quality (recall, MRR, nDCG) for several vector stores on a
small labeled dataset, so you can see which store retrieves best for your data.

Run:  python examples/eval_retrieval.py
"""
import tempfile

from kssrag.core.vectorstores import (
    BM25VectorStore,
    TFIDFVectorStore,
    BM25SVectorStore,
    HybridOfflineVectorStore,
)
from kssrag.eval import compare_retrievers, format_comparison


# A labeled corpus: each document has a stable id in its metadata.
CORPUS = [
    ("d0", "Python is a versatile high level programming language"),
    ("d1", "Machine learning algorithms learn patterns from training data"),
    ("d2", "Deep learning uses neural networks with many layers"),
    ("d3", "The quick brown fox jumps over the lazy dog"),
    ("d4", "Rust provides memory safety without garbage collection"),
    ("d5", "Data science combines statistics and programming skills"),
    ("d6", "Neural networks are inspired by the human brain"),
    ("d7", "Supervised learning requires labeled training examples"),
    ("d8", "Go is a compiled language designed for concurrency"),
    ("d9", "Gradient descent optimizes machine learning model weights"),
]

# Ground truth: query -> set of relevant document ids.
DATASET = [
    ("machine learning training data", {"d1", "d7", "d9"}),
    ("neural networks deep learning", {"d2", "d6"}),
    ("programming languages", {"d0", "d4", "d8", "d5"}),
    ("memory safe systems language", {"d4"}),
]


def build(store_cls):
    docs = [{"content": text, "metadata": {"id": doc_id}} for doc_id, text in CORPUS]
    store = store_cls(persist_path=tempfile.mktemp())
    store.add_documents(docs)
    return store


def main():
    stores = {
        "bm25": build(BM25VectorStore),
        "tfidf": build(TFIDFVectorStore),
        "bm25s": build(BM25SVectorStore),
        "hybrid_offline": build(HybridOfflineVectorStore),
    }
    results = compare_retrievers(
        stores, DATASET, k=5, metrics=["recall", "mrr", "ndcg", "precision"]
    )
    print(format_comparison(results, k=5))


if __name__ == "__main__":
    main()
