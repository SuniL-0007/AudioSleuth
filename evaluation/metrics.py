"""Graded IR metrics over time-span judgments, independent of chunk boundaries."""

from __future__ import annotations

import math
from dataclasses import dataclass

from schemas import Chunk


@dataclass(frozen=True)
class Span:
    """A judged relevant moment in source seconds; grade 2 is primary, 1 is partial."""

    start: float
    end: float
    grade: int
    speaker: str | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end or self.grade not in (1, 2):
            raise ValueError("Spans need a positive interval and a grade of 1 or 2.")


def matches(chunk: Chunk, span: Span, *, min_overlap: float = 0.5) -> bool:
    """Overlap must cover min_overlap of the shorter interval, so either may be longer."""
    if span.speaker is not None and chunk.speaker_id != span.speaker:
        return False
    overlap = min(chunk.end_sec, span.end) - max(chunk.start_sec, span.start)
    shorter = min(chunk.end_sec - chunk.start_sec, span.end - span.start)
    return overlap > 0 and overlap >= min_overlap * shorter


def judge(ranked: list[Chunk], spans: list[Span]) -> tuple[list[int], int]:
    """Return per-rank gains and how many spans were found, crediting each span once.

    A chunk's gain is the best grade among spans it newly covers; a chunk that only
    repeats already-credited spans gains zero, so duplicates cannot inflate scores.
    """
    credited: set[int] = set()
    gains: list[int] = []
    for chunk in ranked:
        new = [i for i, span in enumerate(spans) if i not in credited and matches(chunk, span)]
        credited.update(new)
        gains.append(max((spans[i].grade for i in new), default=0))
    return gains, len(credited)


def precision_at_k(gains: list[int], k: int) -> float:
    """Missing ranks past the end of a short result list count as non-relevant."""
    return sum(1 for gain in gains[:k] if gain > 0) / k


def reciprocal_rank(gains: list[int]) -> float:
    return next((1 / rank for rank, gain in enumerate(gains, 1) if gain > 0), 0.0)


def average_precision(gains: list[int], total_spans: int) -> float:
    if total_spans == 0:
        return 0.0
    hits, total = 0, 0.0
    for rank, gain in enumerate(gains, 1):
        if gain > 0:
            hits += 1
            total += hits / rank
    return total / total_spans


def ndcg_at_k(gains: list[int], spans: list[Span], k: int) -> float:
    """Exponential-gain DCG; the ideal ranking places one span per rank by grade."""

    def dcg(values: list[int]) -> float:
        return sum((2**gain - 1) / math.log2(rank + 1) for rank, gain in enumerate(values, 1))

    ideal = dcg(sorted((span.grade for span in spans), reverse=True)[:k])
    return dcg(gains[:k]) / ideal if ideal > 0 else 0.0


def score_query(ranked: list[Chunk], spans: list[Span], *, k: int = 5) -> dict[str, float]:
    """All metrics for one query's ranked chunks; recall uses spans found within k."""
    gains, _ = judge(ranked, spans)
    _, found_at_k = judge(ranked[:k], spans)
    return {
        "P@1": precision_at_k(gains, 1),
        f"P@{k}": precision_at_k(gains, k),
        f"R@{k}": found_at_k / len(spans) if spans else 0.0,
        "MRR": reciprocal_rank(gains),
        f"nDCG@{k}": ndcg_at_k(gains, spans, k),
        "MAP": average_precision(gains, len(spans)),
    }
