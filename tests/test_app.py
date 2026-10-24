"""Dashboard callbacks, HTML safety, isolated state, and cleanup."""

from pathlib import Path

import gradio as gr
import pytest

from app import (
    MAX_RESULTS,
    backend_status,
    build_app,
    clear_callback,
    highlight,
    load_demo_callback,
    reindex_callback,
    search_callback,
    speaker_color,
    timestamp,
)
from service import CACHE_ROOT, Collection, create_collection, dispose

# State, banners, table, players, filters (8) + result groups + episode, panel, tab, alpha.
COLLECTION_OUTPUTS = 8 + 3 * MAX_RESULTS + 4


def test_html_highlights_escape_untrusted_text() -> None:
    output = highlight('<script>alert("rates")</script> rates & taxes', "rates <script>")
    assert "<script>" not in output
    assert "&lt;" in output and "&amp;" in output
    assert "<mark>rates</mark>" in output
    assert timestamp(61_237) == "01:01.237"


def test_dashboard_constructs_without_neural_dependencies() -> None:
    app = build_app()
    assert len(app.blocks) > 30


def test_search_callback_clip_outputs_and_empty_state(demo_collection: Collection) -> None:
    outputs = search_callback("heated rates", 0.65, "All speakers", 0, 0, demo_collection)
    assert len(outputs) == 1 + 3 * MAX_RESULTS
    assert "moments found" in outputs[0]
    assert Path(outputs[3]).is_file()
    with pytest.raises(gr.Error):
        search_callback("test", 0.5, "All speakers", 0, 0, None)
    empty = search_callback("rates", 0.5, "missing", 0, 0, demo_collection)
    assert "No matches" in empty[0]
    assert all(empty[i] is None for i in range(3, 1 + 3 * MAX_RESULTS, 3))


def test_collections_are_independent_and_cleanup_is_scoped() -> None:
    first, second = create_collection(), create_collection()
    try:
        assert first.directory != second.directory
        assert first.index is not second.index
        first.index.chunks.clear()
        assert len(second.index.chunks) > 0
        dispose(first)
        assert not first.directory.exists()
        assert second.directory.exists()
    finally:
        dispose(first)
        dispose(second)


def test_demo_button_replaces_only_its_own_previous_collection() -> None:
    old = create_collection()
    outputs = load_demo_callback(old)
    new = outputs[0]
    try:
        assert isinstance(new, Collection)
        assert len(outputs) == COLLECTION_OUTPUTS
        assert not old.directory.exists()
        assert new.directory.parent == CACHE_ROOT
    finally:
        dispose(new)


def test_reindex_preserves_transcript_source_and_speakers() -> None:
    old = create_collection()
    words = old.recording.words.copy()
    source_hash = old.recording.source_sha256
    result = reindex_callback(False, old)
    current = result[0]
    try:
        assert len(result) == COLLECTION_OUTPUTS
        assert current.recording.words == words
        assert current.recording.source_sha256 == source_hash
        assert not old.directory.exists()
        assert current.directory.exists()
        assert len(current.index.chunks) == 13
    finally:
        dispose(current)


def test_clear_resets_players_upload_and_only_current_collection() -> None:
    current, other = create_collection(), create_collection()
    try:
        result = clear_callback(current)
        assert len(result) == COLLECTION_OUTPUTS + 1
        assert result[0] is None and result[-1] is None
        assert result[3] == [] and result[4] is None and result[6] is None
        assert not current.directory.exists()
        assert other.directory.exists()
        assert all(result[i] is None for i in range(10, 8 + 3 * MAX_RESULTS, 3))
        assert len(clear_callback(None)) == COLLECTION_OUTPUTS + 1
    finally:
        dispose(current)
        dispose(other)


def test_speaker_colors_are_stable_and_unknown_is_neutral() -> None:
    assert speaker_color("SPEAKER_00") == "mint"
    assert speaker_color("SPEAKER_01") == "violet"
    assert speaker_color("SPEAKER_05") == speaker_color("SPEAKER_01")
    assert speaker_color("UNKNOWN") == "neutral"


def test_backend_status_never_displays_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "test-token-not-for-display")
    status = backend_status()
    assert "Configured" in status
    assert "test-token-not-for-display" not in status
