"""Hybrid ranking with selectable fusion, stable ties, and pre-ranking metadata filters."""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

from indexing import DualIndex
from schemas import AudioSleuthError, SearchHit, Vectors

TextScorer = Literal["vectors", "bm25"]
Fusion = Literal["linear", "zscore", "rrf"]
RRF_K = 60  # Cormack et al. (2009); damps the influence of top ranks.


def _zscore(values: Vectors) -> Vectors:
    """Standardize over the filtered candidates; a constant modality carries no ordering."""
    spread = float(values.std())
    if spread < 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / spread


def _reciprocal_ranks(values: Vectors, starts: list[int]) -> Vectors:
    """1/(k + rank) with the same stable start-time tie-breaking as the final order."""
    order = sorted(range(len(values)), key=lambda i: (-float(values[i]), starts[i]))
    ranks = np.empty(len(values), dtype=np.float32)
    ranks[order] = np.arange(1, len(values) + 1)
    return np.asarray(1 / (RRF_K + ranks), dtype=np.float32)


def search(
    index: DualIndex,
    query: str,
    *,
    alpha: float = 0.65,
    speaker: str | None = None,
    episode: str | None = None,
    min_confidence: float = 0.0,
    min_score: float = -1.0,
    top_k: int = 5,
    text_scorer: TextScorer = "vectors",
    fusion: Fusion = "linear",
) -> list[SearchHit]:
    """Rank candidates by a weighted fusion of text and audio evidence, not a probability.

    linear: alpha*cos(text) + (1-alpha)*cos(audio), the original cosine identity.
    zscore: the same weights over per-modality standardized scores.
    rrf:    weighted reciprocal rank fusion, which ignores score scales entirely.
    A modality without query evidence contributes nothing to zscore or rrf.
    min_score applies only to linear fusion, whose scores lie in [-1, 1].
    episode filters by source SHA-256; speaker and confidence filter before ranking.
    """
    if not query.strip() or len(query) > 1000:
        raise AudioSleuthError("Enter a query of 1 to 1,000 characters.")
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise AudioSleuthError("Semantic weight must be between zero and one.")
    if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
        raise AudioSleuthError("Alignment confidence must be between zero and one.")
    if not math.isfinite(min_score) or not -1 <= min_score <= 1 or not 1 <= top_k <= 20:
        raise AudioSleuthError("Use a score threshold in [-1, 1] and top_k in [1, 20].")
    if text_scorer not in ("vectors", "bm25") or fusion not in ("linear", "zscore", "rrf"):
        raise AudioSleuthError("Unknown text scorer or fusion method.")
    if text_scorer == "bm25" and fusion == "linear" and 0 < alpha < 1:
        raise AudioSleuthError(
            "BM25 scores are unbounded; combine them with audio using z-score or RRF fusion."
        )
    candidates = [
        i
        for i, chunk in enumerate(index.chunks)
        if (speaker is None or chunk.speaker_id == speaker)
        and (episode is None or chunk.source_sha256 == episode)
        and chunk.confidence >= min_confidence
    ]
    if not candidates:
        return []
    use_bm25 = text_scorer == "bm25" and alpha > 0
    text, audio = index.encode_query(
        query, include_text=alpha > 0 and not use_bm25, include_audio=alpha < 1
    )
    picked = np.asarray(candidates)
    if use_bm25:
        if index.bm25 is None:
            raise AudioSleuthError("This index has no BM25 scorer; re-index the recording.")
        text_scores = index.bm25.scores(query)
    else:
        text_scores = np.clip(index.text_vectors @ text, -1, 1)
    audio_scores = np.clip(index.audio_vectors @ audio, -1, 1)
    # Evidence is judged among the filtered candidates: a term found only in another
    # episode or speaker's chunks must not surface arbitrary zero-score matches here.
    text_evidence = (use_bm25 or bool(np.any(text))) and bool(np.any(text_scores[picked]))
    audio_evidence = bool(np.any(audio)) and bool(np.any(audio_scores[picked]))
    if not text_evidence and not audio_evidence:
        return []
    starts = [index.chunks[i].start_ms for i in candidates]
    fused = np.zeros(len(candidates), dtype=np.float32)
    for weight, scores, evidence in (
        (alpha, text_scores, text_evidence),
        (1 - alpha, audio_scores, audio_evidence),
    ):
        if fusion == "linear":
            fused += weight * scores[picked]
        elif weight > 0 and evidence:
            local = scores[picked]
            normalized = _zscore(local) if fusion == "zscore" else _reciprocal_ranks(local, starts)
            fused += weight * normalized
    ordered = sorted(range(len(candidates)), key=lambda j: (-float(fused[j]), starts[j]))
    return [
        SearchHit(
            index.chunks[candidates[j]],
            float(fused[j]),
            float(text_scores[candidates[j]]),
            float(audio_scores[candidates[j]]),
        )
        for j in ordered
        if fusion != "linear" or fused[j] >= min_score
    ][:top_k]
