"""Okapi BM25 over chunk transcripts, scored as one sparse matrix-vector product."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from schemas import Vectors


class BM25:
    """Precomputed per-term document weights; a query's score sums its matched terms.

    Scores are unbounded relevance evidence, not cosines, so they must not be
    linearly mixed with cosine similarities without normalization.
    """

    def __init__(self, texts: list[str], *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 < 0 or not 0 <= b <= 1:
            raise ValueError("BM25 needs k1 >= 0 and b in [0, 1].")
        self.size = len(texts)
        self.vectorizer = CountVectorizer(
            lowercase=True, stop_words="english", token_pattern=r"(?u)\b\w+\b"
        )
        try:
            counts = sparse.csr_matrix(self.vectorizer.fit_transform(texts), dtype=np.float64)
        except ValueError:
            self.weights: sparse.csr_matrix | None = None  # Only stopwords: no lexical evidence.
            return
        lengths = np.asarray(counts.sum(axis=1)).ravel()
        frequency = np.bincount(counts.indices, minlength=counts.shape[1])
        idf = np.log1p((self.size - frequency + 0.5) / (frequency + 0.5))
        norm = k1 * (1 - b + b * lengths / max(float(lengths.mean()), 1e-12))
        rows = np.repeat(np.arange(self.size), np.diff(counts.indptr))
        tf = counts.data
        counts.data = idf[counts.indices] * tf * (k1 + 1) / (tf + norm[rows])
        self.weights = counts

    def scores(self, query: str) -> Vectors:
        """Each query term counts once; out-of-vocabulary queries score zero everywhere."""
        if self.weights is None:
            return np.zeros(self.size, dtype=np.float32)
        terms = self.vectorizer.transform([query])
        terms.data[:] = 1
        return np.asarray((self.weights @ terms.T).toarray().ravel(), dtype=np.float32)
