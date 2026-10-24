"""Hand-computed metric values, span judging rules, and an end-to-end demo evaluation."""

from __future__ import annotations

import math

import pytest

from evaluation.metrics import (
    Span,
    average_precision,
    judge,
    matches,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    score_query,
)
from evaluation.run_eval import CONFIGS, QRELS, load_qrels, report, run
from schemas import AudioSleuthError, Chunk


def chunk(start: float, end: float, speaker: str = "SPEAKER_00") -> Chunk:
    return Chunk("c", speaker, int(start * 1000), int(end * 1000), "t", 1.0, 1.0, "aligned", "", "")


def test_overlap_uses_shorter_interval_and_optional_speaker() -> None:
    span = Span(10, 12, 2)
    assert matches(chunk(0, 20), span)  # Span fully inside a long chunk.
    assert matches(chunk(11, 15), span)  # 1 s of overlap is half the 2 s span.
    assert not matches(chunk(11.5, 15), span)
    assert not matches(chunk(12, 15), span)  # Touching endpoints do not overlap.
    assert not matches(chunk(0, 20, "SPEAKER_01"), Span(10, 12, 2, "SPEAKER_00"))


def test_each_span_is_credited_once() -> None:
    spans = [Span(0, 5, 2), Span(10, 15, 1)]
    gains, found = judge([chunk(0, 5), chunk(1, 4), chunk(10, 15), chunk(30, 40)], spans)
    assert gains == [2, 0, 1, 0]
    assert found == 2
    gains, found = judge([chunk(0, 15)], spans)  # One chunk covering both earns the best grade.
    assert gains == [2]
    assert found == 2


def test_hand_computed_metrics() -> None:
    gains = [0, 2, 0, 1]
    spans = [Span(0, 1, 2), Span(2, 3, 1), Span(4, 5, 1)]
    assert precision_at_k(gains, 1) == 0
    assert precision_at_k(gains, 4) == 0.5
    assert precision_at_k([2], 5) == 0.2  # Short result lists are padded with misses.
    assert reciprocal_rank(gains) == 0.5
    assert reciprocal_rank([0, 0]) == 0
    assert average_precision(gains, 3) == pytest.approx((1 / 2 + 2 / 4) / 3)
    dcg = 3 / math.log2(3) + 1 / math.log2(5)
    ideal = 3 / math.log2(2) + 1 / math.log2(3) + 1 / math.log2(4)
    assert ndcg_at_k(gains, spans, 4) == pytest.approx(dcg / ideal)
    assert ndcg_at_k([2, 1, 1], spans, 3) == pytest.approx(1.0)


def test_recall_counts_only_spans_found_within_cutoff() -> None:
    spans = [Span(0, 5, 2), Span(10, 15, 2)]
    metrics = score_query([chunk(0, 5), chunk(20, 25), chunk(10, 15)], spans, k=2)
    assert metrics["R@2"] == 0.5
    assert metrics["MRR"] == 1.0
    assert metrics["MAP"] == pytest.approx((1 + 2 / 3) / 2)


def test_invalid_spans_and_qrels_rejected() -> None:
    with pytest.raises(ValueError):
        Span(5, 5, 2)
    with pytest.raises(ValueError):
        Span(0, 1, 3)
    assert len(load_qrels(QRELS)) >= 15


def test_demo_evaluation_reports_bounded_metrics() -> None:
    results = run()
    assert len(results) == len(CONFIGS)
    for result in results:
        for row in result["queries"]:
            values = [value for key, value in row.items() if key not in ("id", "type")]
            assert all(0 <= value <= 1 for value in values)
    text_only = next(result for result in results if result["label"] == "text only")
    lexical = [row["MRR"] for row in text_only["queries"] if row["type"] == "lexical"]
    assert sum(lexical) / len(lexical) > 0.5  # TF-IDF must find literal topic matches.
    assert "### lexical queries" in report(results)


def test_neural_run_refuses_silent_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without the ML stack, --neural must fail loudly rather than duplicate offline rows."""
    import builtins

    real_import = builtins.__import__

    def blocked(name: str, *args: object, **kwargs: object) -> object:
        if name.split(".")[0] in ("sentence_transformers", "transformers"):
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(AudioSleuthError, match="fell back"):
        run(neural=True)
