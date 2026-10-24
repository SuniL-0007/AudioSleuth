"""Multi-episode collections: shared index, episode filter, provenance, and UI wiring."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

import service
from app import MAX_RESULTS, add_callback, search_callback
from retrieval import search
from schemas import AudioSleuthError, Recording
from service import Collection, add_to_collection, create_collection, dispose


def second_episode(demo: Recording) -> Recording:
    """The demo's first 45 seconds under a distinct hash, standing in for an ASR upload."""
    words = [word for word in demo.words if word.end <= 45]
    return replace(
        demo,
        samples=demo.samples[: 45 * 16_000].copy(),
        source_name="Second episode",
        source_sha256="a" * 64,
        words=words,
        intervals=[i for i in demo.intervals if i.end <= 45],
        mode="Test fixture",
    )


@pytest.fixture
def pair(monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Collection, Collection]]:
    first = create_collection()
    monkeypatch.setattr(service, "ingest_upload", lambda _: second_episode(first.recording))
    both = add_to_collection(first, upload=Path("second.wav"))
    yield first, both
    dispose(first)
    dispose(both)


def test_added_episode_shares_one_index(pair: tuple[Collection, Collection]) -> None:
    first, both = pair
    assert [r.source_name for r in both.recordings][1] == "Second episode"
    assert len(both.index.chunks) > len(first.index.chunks)
    assert len({chunk.chunk_id for chunk in both.index.chunks}) == len(both.index.chunks)
    assert both.index.text_backend.startswith("TF-IDF")  # Rebuilt, not the bundled vectors.
    assert all(path.is_file() for path in both.source_paths.values())
    assert len(first.index.chunks) == 13  # The previous collection is left untouched.


def test_episode_filter_and_cross_episode_ranking(pair: tuple[Collection, Collection]) -> None:
    _, both = pair
    hits = search(both.index, "interest rates", top_k=20)
    assert {hit.chunk.source_sha256 for hit in hits} == {r.source_sha256 for r in both.recordings}
    only = search(both.index, "interest rates", episode="a" * 64, top_k=20)
    assert only and all(hit.chunk.source_sha256 == "a" * 64 for hit in only)
    assert search(both.index, "battery storage", episode="a" * 64) == []  # Said after 45 s.


def test_manifest_lists_every_source(pair: tuple[Collection, Collection]) -> None:
    _, both = pair
    manifest = json.loads(both.manifest_path.read_text(encoding="utf-8"))
    assert [s["source_sha256"] for s in manifest["sources"]] == [
        r.source_sha256 for r in both.recordings
    ]
    assert len(manifest["chunks"]) == len(both.index.chunks)


def test_duplicate_episode_is_rejected(
    pair: tuple[Collection, Collection], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, both = pair
    with pytest.raises(AudioSleuthError, match="already"):
        add_to_collection(both)  # The demo itself is already present.


def test_add_callback_outputs_and_episode_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = create_collection()
    monkeypatch.setattr(service, "ingest_upload", lambda _: second_episode(first.recording))
    outputs = add_callback("second.wav", False, first)
    both = outputs[0]
    try:
        assert len(outputs) == 8 + 3 * MAX_RESULTS + 4
        assert not first.directory.exists()
        assert outputs[-4]["choices"][1:] == both.episodes()
        results = search_callback(
            "interest rates", 1.0, "All speakers", 0, 0, both, "a" * 64, "bm25", "rrf"
        )
        assert "reciprocal rank fusion" in results[0]
        assert "Second episode" in results[2]
    finally:
        dispose(first)
        dispose(both)


def test_add_keeps_neural_mode_of_current_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """An unchecked box must not silently rebuild a neural collection with offline encoders."""
    first = create_collection()
    first.index.text_backend = "sentence-transformers"  # Pretend the current index is neural.
    seen: list[bool] = []
    real_build = service.build_index

    def recording_build(chunks: list, *, neural: bool = False):  # type: ignore[no-untyped-def]
        seen.append(neural)
        return real_build(chunks, neural=False)  # Avoid loading real models in tests.

    monkeypatch.setattr(service, "ingest_upload", lambda _: second_episode(first.recording))
    monkeypatch.setattr(service, "build_index", recording_build)
    both = add_to_collection(first, upload=Path("second.wav"))
    try:
        assert seen == [True]
    finally:
        dispose(first)
        dispose(both)
