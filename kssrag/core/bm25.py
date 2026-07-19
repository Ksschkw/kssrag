"""
Self-contained BM25 Okapi implementation.

BM25 is the standard lexical relevance function used by search engines. This is
a dependency-free implementation owned by KSS RAG so the lexical retrieval path
is fully under our control (tokenization, scoring, tie-breaking) rather than
delegated to a third-party library.

Scoring, for a query Q against document D in a corpus of N documents:

    score(D, Q) = sum over q in Q of  IDF(q) * ( f(q,D) * (k1 + 1) )
                                       / ( f(q,D) + k1 * (1 - b + b * |D| / avgdl) )

    IDF(q) = ln( 1 + (N - n(q) + 0.5) / (n(q) + 0.5) )

where f(q,D) is the term frequency of q in D, |D| is the document length in
tokens, avgdl is the average document length, n(q) is the number of documents
containing q, and (k1, b) are the standard tuning parameters.

Note on the IDF variant: the `1 +` inside the log is the Lucene/Elasticsearch
form, which keeps IDF non-negative for every term. rank-bm25's BM25Okapi uses
the bare `ln((N - n + 0.5)/(n + 0.5))` (which can be negative and applies an
epsilon floor). The two differ by a per-term additive constant, so document
*rankings* are identical — verified against rank-bm25 in the benchmark — even
though absolute scores differ. Since retrieval and RRF fusion depend only on
rank order, this choice is intentional.
"""
import math
import re
from collections import Counter
from typing import Dict, List

_TOKEN_RE = re.compile(r"\w+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenization (matches the BM25VectorStore tokenizer)."""
    return _TOKEN_RE.findall(text.lower())


class BM25Okapi:
    """BM25 Okapi ranking over a fixed tokenized corpus."""

    def __init__(self, corpus_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus_tokens)
        self.doc_freqs: List[Counter] = []
        self.doc_lengths: List[int] = []
        self.idf: Dict[str, float] = {}

        total_len = 0
        term_doc_count: Dict[str, int] = {}
        for tokens in corpus_tokens:
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for term in freqs:
                term_doc_count[term] = term_doc_count.get(term, 0) + 1

        self.avgdl = (total_len / self.corpus_size) if self.corpus_size else 0.0
        self._compute_idf(term_doc_count)

    def _compute_idf(self, term_doc_count: Dict[str, int]) -> None:
        for term, n_q in term_doc_count.items():
            # Probabilistic IDF with +1 inside the log to keep it non-negative.
            self.idf[term] = math.log(1 + (self.corpus_size - n_q + 0.5) / (n_q + 0.5))

    def get_scores(self, query_tokens: List[str]) -> List[float]:
        """Return a BM25 score for every document in the corpus."""
        scores = [0.0] * self.corpus_size
        k1, b, avgdl = self.k1, self.b, self.avgdl
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freqs in enumerate(self.doc_freqs):
                f = freqs.get(term)
                if not f:
                    continue
                denom = f + k1 * (1 - b + b * self.doc_lengths[i] / avgdl)
                scores[i] += idf * (f * (k1 + 1)) / denom
        return scores
