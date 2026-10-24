"""BM25 arithmetic, fusion semantics, and the episode filter."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bm25 import BM25
from retrieval import RRF_K, search
from schemas import AudioSleuthError
from service import Collection


def test_bm25_matches_hand_computed_okapi_score() -> None:
    scorer = BM25(["apple banana", "apple apple cherry", "banana"])
    idf = math.log1p((3 - 1 + 0.5) / (1 + 0.5))
    norm = 1.5 * (1 - 0.75 + 0.75 * 3 / 2)  # Document length 3, average length 2.
    expected = idf * 1 * 2.5 / (1 + norm)
    assert scorer.scores("cherry") == pytest.approx([0, expected, 0], rel=1e-6)
    assert scorer.scores("Cherry cherry") == pytest.approx([0, expected, 0], rel=1e-6)


def test_bm25_without_lexical_evidence_scores_zero() -> None:
    assert not BM25(["apple banana"]).scores("the and of").any()
    assert not BM25(["the and of"]).scores("apple").any()  # Stopword-only corpus.
    with pytest.raises(ValueError):
        BM25(["apple"], b=2)


def test_bm25_text_search_and_linear_mixing_guard(demo_collection: Collection) -> None:
    hits = search(demo_collection.index, "battery storage", alpha=1, text_scorer="bm25")
    assert "battery storage" in hits[0].chunk.transcript.lower()
    assert hits[0].score == hits[0].text_score > 0
    with pytest.raises(AudioSleuthError, match="unbounded"):
        search(demo_collection.index, "heated", alpha=0.5, text_scorer="bm25")
    with pytest.raises(AudioSleuthError):
        search(demo_collection.index, "heated", fusion="max")  # type: ignore[arg-type]


def test_rrf_scores_are_weighted_reciprocal_ranks(demo_collection: Collection) -> None:
    hits = search(demo_collection.index, "valuation", alpha=1, fusion="rrf", top_k=3)
    assert [hit.score for hit in hits] == pytest.approx([1 / (RRF_K + r) for r in (1, 2, 3)])


@pytest.mark.parametrize("fusion", ["zscore", "rrf"])
def test_modality_without_evidence_does_not_reorder(
    demo_collection: Collection, fusion: str
) -> None:
    """'valuation' has no tone words, so offline audio must not perturb the text order."""
    text_only = search(demo_collection.index, "valuation", alpha=1, top_k=10)
    fused = search(demo_collection.index, "valuation", alpha=0.5, fusion=fusion, top_k=10)  # type: ignore[arg-type]
    assert [hit.chunk.chunk_id for hit in fused] == [hit.chunk.chunk_id for hit in text_only]


def test_zscore_scores_are_standardized(demo_collection: Collection) -> None:
    hits = search(demo_collection.index, "heated", alpha=0, fusion="zscore", top_k=20)
    scores = np.array([hit.score for hit in hits])
    assert len(hits) == len(demo_collection.index.chunks)
    assert scores.mean() == pytest.approx(0, abs=1e-5)
    assert scores.std() == pytest.approx(1, abs=1e-5)


def test_episode_filter_applies_before_ranking(demo_collection: Collection) -> None:
    episode = demo_collection.recording.source_sha256
    assert search(demo_collection.index, "valuation", episode=episode)
    assert search(demo_collection.index, "valuation", episode="0" * 64) == []
