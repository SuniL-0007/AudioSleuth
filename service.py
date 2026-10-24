"""Session-local collection lifecycle and an auditable provenance manifest."""

from __future__ import annotations

import json
import logging
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import soundfile as sf

from chunking import chunk_recording
from indexing import DualIndex, build_index, load_demo_index
from ingestion import ASSETS, ingest_upload, load_demo
from schemas import SAMPLE_RATE, AudioSleuthError, Recording

LOGGER = logging.getLogger(__name__)
CACHE_ROOT = Path(tempfile.gettempdir()) / "audiosleuth-collections"


@dataclass
class Collection:
    """One browser session's episodes, index, and files, never a shared global corpus."""

    recordings: list[Recording]
    index: DualIndex
    directory: Path
    source_paths: dict[str, Path]
    manifest_path: Path

    @property
    def recording(self) -> Recording:
        """The most recently added episode, which the source player shows."""
        return self.recordings[-1]

    @property
    def source_path(self) -> Path:
        return self.source_paths[self.recording.source_sha256]

    def episodes(self) -> list[tuple[str, str]]:
        """(display name, source SHA-256) pairs in the order episodes were added."""
        return [
            (f"{r.source_name} [{r.source_sha256[:8]}]", r.source_sha256) for r in self.recordings
        ]


def _remove_workspace(directory: Path) -> None:
    resolved = directory.resolve()
    if resolved.parent != CACHE_ROOT.resolve() or not resolved.name.startswith("collection-"):
        raise ValueError("Refusing to remove a directory not owned by this collection.")
    if resolved.exists():
        shutil.rmtree(resolved)


def dispose(collection: Collection | None) -> None:
    """Delete only the named collection workspace on expiry or explicit replacement."""
    if collection is not None:
        _remove_workspace(collection.directory)


def _build_collection(
    recordings: list[Recording], *, neural: bool, precomputed: bool = False
) -> Collection:
    """Build in a fresh owned workspace and remove partial files after any failure.

    Every episode is chunked into one shared index, so IDF statistics and score
    normalization span the whole collection rather than a single recording.
    """
    hashes = [recording.source_sha256 for recording in recordings]
    if not recordings or len(set(hashes)) != len(hashes):
        raise AudioSleuthError("A collection needs distinct recordings; this one is present.")
    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="collection-", dir=CACHE_ROOT))
    completed = False
    try:
        chunks = [chunk for r in recordings for chunk in chunk_recording(r, directory)]
        index = (
            load_demo_index(chunks, ASSETS)
            if precomputed and not neural and len(recordings) == 1
            else build_index(chunks, neural=neural)
        )
        source_paths: dict[str, Path] = {}
        for recording in recordings:
            path = directory / f"source-{recording.source_sha256[:12]}.wav"
            sf.write(path, recording.samples, SAMPLE_RATE, subtype="FLOAT")
            source_paths[recording.source_sha256] = path
        manifest_path = directory / "provenance.json"
        manifest = {
            "sources": [
                {
                    "source_name": recording.source_name,
                    "source_sha256": recording.source_sha256,
                    "mode": recording.mode,
                    "duration_sec": recording.duration,
                    "notices": recording.notices,
                }
                for recording in recordings
            ],
            "sample_rate": SAMPLE_RATE,
            "text_backend": index.text_backend,
            "audio_backend": index.audio_backend,
            "score_meaning": "Hybrid relevance evidence, not calibrated probability.",
            "notices": index.notices,
            "chunks": [chunk.payload() for chunk in chunks],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        collection = Collection(recordings, index, directory, source_paths, manifest_path)
        completed = True
        return collection
    finally:
        if not completed:
            _remove_workspace(directory)


def create_collection(*, upload: Path | None = None, neural: bool = False) -> Collection:
    """Construct atomically; a failed ingestion never replaces the previous collection."""
    recording = load_demo() if upload is None else ingest_upload(upload)
    return _build_collection([recording], neural=neural, precomputed=upload is None)


def add_to_collection(
    current: Collection, *, upload: Path | None = None, neural: bool = False
) -> Collection:
    """Add an episode (or the demo) and rebuild one index; the old collection is untouched."""
    recording = load_demo() if upload is None else ingest_upload(upload)
    if any(r.source_sha256 == recording.source_sha256 for r in current.recordings):
        raise AudioSleuthError(f"{recording.source_name} is already in this collection.")
    return _build_collection(current.recordings + [recording], neural=neural)


def reindex_collection(current: Collection, *, neural: bool) -> Collection:
    """Change encoders without retranscribing or relabeling any episode."""
    return _build_collection(current.recordings, neural=neural)
