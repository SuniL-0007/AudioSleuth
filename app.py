"""AudioSleuth: local, session-isolated multimodal podcast retrieval dashboard."""

from __future__ import annotations

import argparse
import html
import importlib.util
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr  # noqa: E402

from indexing import acoustic_query  # noqa: E402
from retrieval import search  # noqa: E402
from schemas import AudioSleuthError, SearchHit  # noqa: E402
from service import (  # noqa: E402
    Collection,
    add_to_collection,
    create_collection,
    dispose,
    is_neural,
    reindex_collection,
)

ROOT = Path(__file__).parent
MAX_RESULTS = 10
ALL_EPISODES = "All episodes"
SPEAKER_COLORS = ("mint", "violet", "amber", "sky")
# Default text weight per backend, from the demo evaluation (evaluation/run_eval.py):
# offline MRR is flat for alpha in [0.5, 0.95]; neural MRR rises with alpha because CLAP
# adds little, so 0.9 keeps a small audio say for tone queries.
DEFAULT_ALPHA = {"offline": 0.65, "neural": 0.9}
SCORERS = [
    ("Index vectors (TF-IDF or MiniLM)", "vectors"),
    ("BM25 (Okapi, stopwords removed)", "bm25"),
]
FUSIONS = [
    ("Linear cosine (original)", "linear"),
    ("Z-score normalized", "zscore"),
    ("Reciprocal rank fusion", "rrf"),
]
FUSION_LABELS = {
    "linear": "hybrid cosine relevance",
    "zscore": "z-score fusion",
    "rrf": "reciprocal rank fusion",
}


def backend_status() -> str:
    """Report installation prerequisites without loading models or exposing credentials."""
    rows = [
        "| Component | Local prerequisite |",
        "|---|---|",
    ]
    for name, module in (
        ("WhisperX ASR", "whisperx"),
        ("Sentence-transformers", "sentence_transformers"),
        ("LAION-CLAP via Transformers", "transformers"),
    ):
        installed = importlib.util.find_spec(module) is not None
        rows.append(f"| {name} | {'Installed' if installed else 'Not installed'} |")
    rows.append(f"| Diarization token | {'Configured' if os.getenv('HF_TOKEN') else 'Missing'} |")
    rows.append(f"| FFmpeg executable | {'Found' if shutil.which('ffmpeg') else 'Not found'} |")
    return "\n".join(rows) + (
        "\n\nInstalled does not mean model weights are cached or imports are compatible. "
        "The bundled offline demo needs none of these optional prerequisites."
    )


def timestamp(milliseconds: int) -> str:
    """Format the precise indexed time boundary without rounding to whole seconds."""
    minutes, remaining = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remaining, 1000)
    return f"{minutes:02d}:{seconds:02d}.{millis:03d}"


def highlight(transcript: str, query: str) -> str:
    """Escape all untrusted text before adding bounded query-word highlight markup."""
    terms = set(re.findall(r"\w{3,}", query, flags=re.UNICODE))
    if not terms:
        return html.escape(transcript)
    pattern = re.compile(r"\b(" + "|".join(re.escape(t) for t in sorted(terms)) + r")\b", re.I)
    result: list[str] = []
    last = 0
    for match in pattern.finditer(transcript):
        result.extend(
            (
                html.escape(transcript[last : match.start()]),
                f"<mark>{html.escape(match.group())}</mark>",
            )
        )
        last = match.end()
    result.append(html.escape(transcript[last:]))
    return "".join(result)


def speaker_color(speaker_id: str) -> str:
    """Stable palette slot from the label's trailing number; UNKNOWN stays neutral."""
    digits = re.search(r"(\d+)$", speaker_id)
    if digits is None:
        return "neutral"
    return SPEAKER_COLORS[int(digits.group(1)) % len(SPEAKER_COLORS)]


def hit_card(
    hit: SearchHit,
    query: str,
    rank: int,
    *,
    alpha: float = 0.65,
    scorer: str = "vectors",
    episode: str = "",
) -> str:
    """Display relevance separately from timing confidence and speaker overlap."""
    chunk = hit.chunk
    color = speaker_color(chunk.speaker_id)
    text_label = "Text BM25" if scorer == "bm25" else "Text cosine"
    text_score = f"{hit.text_score:.3f}" if alpha > 0 else "not evaluated"
    audio_score = f"{hit.audio_score:.3f}" if alpha < 1 else "not evaluated"
    return f"""
    <article class="hit">
      <div class="hit-top">
        <span class="rank">MATCH {rank:02d}</span>
        <span class="speaker {color}">{html.escape(chunk.speaker_id)}</span>
        <span class="timecode">{timestamp(chunk.start_ms)} &rarr; {timestamp(chunk.end_ms)}</span>
        <span class="score">{hit.score:.3f}<small> relevance</small></span>
      </div>
      <p class="quote">{highlight(chunk.transcript, query)}</p>
      <div class="hit-meta">
        <span>{text_label} <b>{text_score}</b></span>
        <span>Audio cosine <b>{audio_score}</b></span>
        <span>Alignment quality <b>{chunk.confidence:.2f}</b></span>
        <span>Speaker overlap <b>{chunk.speaker_confidence:.2f}</b></span>
        <span>{html.escape(chunk.timing)} timing</span>
      </div>
      <div class="provenance">{html.escape(episode)}
        &nbsp; / &nbsp; SOURCE SHA-256 {chunk.source_sha256[:16]}...
        &nbsp; / &nbsp; {html.escape(chunk.chunk_id)}
        &nbsp; / &nbsp; {(chunk.end_ms - chunk.start_ms) / 1000:.3f}s clip</div>
    </article>"""


def _empty_results() -> list[Any]:
    return [
        component for _ in range(MAX_RESULTS) for component in (gr.update(visible=False), "", None)
    ]


def _short(label: str, limit: int = 28) -> str:
    """Keep the episode column narrow; the hash suffix stays visible for disambiguation."""
    name, _, digest = label.rpartition(" [")
    if len(label) <= limit or not name:
        return label
    return f"{name[: limit - 12].rstrip()}… [{digest}"


def _collection_outputs(collection: Collection, elapsed: float) -> list[Any]:
    """Outputs for every index-building action; also reveals the loaded panel and Search tab."""
    recordings, index = collection.recordings, collection.index
    speakers = sorted({chunk.speaker_id for chunk in index.chunks})
    names = {sha: name for name, sha in collection.episodes()}
    minutes = sum(recording.duration for recording in recordings) / 60
    plural = "s" if len(recordings) != 1 else ""
    stats = f"""
    <div class="metrics">
      <div><strong>{minutes:.1f}<small> min</small></strong>
      <span>{len(recordings)} episode{plural}</span></div>
      <div><strong>{len(speakers)}</strong><span>Speaker labels</span></div>
      <div><strong>{len(index.chunks)}</strong><span>Retrievable moments</span></div>
      <div><strong>{elapsed:.2f}<small> s</small></strong><span>Index load / build</span></div>
    </div>"""
    notices = "".join(
        f"<li>{html.escape(notice)}</li>"
        for notice in [n for r in recordings for n in r.notices] + index.notices
    )
    sources = "<br>".join(
        f"{html.escape(r.source_name)} &middot; {html.escape(r.mode)}" for r in recordings
    )
    status = f"""
    <div class="status-ready"><b>INDEX READY</b> &nbsp; {len(recordings)} episode{plural}
    <p>{sources}<br>{html.escape(index.text_backend)}
    &middot; {html.escape(index.audio_backend)}</p>
    <details open><summary>Evidence &amp; limitations</summary><ul>{notices}</ul>
    <p>Times address exact source slices; alignment accuracy is not guaranteed to a millisecond.
    Relevance is not a probability. Synthetic alignment quality 1.0 means scripted ground truth.</p>
    </details></div>"""
    table = [
        [
            _short(names[chunk.source_sha256]),
            chunk.speaker_id,
            timestamp(chunk.start_ms),
            timestamp(chunk.end_ms),
            round(chunk.confidence, 3),
            chunk.timing,
            chunk.transcript,
        ]
        for chunk in index.chunks
    ]
    return [
        collection,
        stats,
        status,
        table,
        str(collection.source_path),
        gr.update(choices=["All speakers"] + speakers, value="All speakers"),
        str(collection.manifest_path),
        "<div class='search-summary'>Index ready. Search for a topic, a tone, or both.</div>",
        *_empty_results(),
        gr.update(choices=[ALL_EPISODES, *collection.episodes()], value=ALL_EPISODES),
        gr.update(visible=True),
        gr.Tabs(selected="search"),
        gr.update(value=DEFAULT_ALPHA["neural" if is_neural(index) else "offline"]),
    ]


def load_demo_callback(current: Collection | None) -> list[Any]:
    """The one-click demo never loads model weights, accesses a token, or downloads data."""
    start = time.perf_counter()
    try:
        collection = create_collection()
    except AudioSleuthError as exc:
        raise gr.Error(str(exc)) from exc
    outputs = _collection_outputs(collection, time.perf_counter() - start)
    dispose(current)
    return outputs


def process_callback(upload: str | None, neural: bool, current: Collection | None) -> list[Any]:
    """Preserve previous results on failure; a missing backend never fabricates uploads."""
    if upload is None:
        raise gr.Error("Choose a WAV or MP3 file, or use the pre-indexed demo.")
    start = time.perf_counter()
    try:
        collection = create_collection(upload=Path(upload), neural=neural)
    except AudioSleuthError as exc:
        raise gr.Error(str(exc)) from exc
    outputs = _collection_outputs(collection, time.perf_counter() - start)
    dispose(current)
    return outputs


def add_callback(upload: str | None, neural: bool, current: Collection | None) -> list[Any]:
    """Grow the session corpus by one episode; adding to nothing starts a new collection."""
    if current is None:
        return process_callback(upload, neural, current)
    if upload is None:
        raise gr.Error("Choose a WAV or MP3 file to add to the current collection.")
    start = time.perf_counter()
    try:
        collection = add_to_collection(current, upload=Path(upload))  # Keeps current mode.
    except AudioSleuthError as exc:
        raise gr.Error(str(exc)) from exc
    outputs = _collection_outputs(collection, time.perf_counter() - start)
    dispose(current)
    return outputs


def reindex_callback(neural: bool, current: Collection | None) -> list[Any]:
    """Let the demo exercise real neural encoders without running ASR again."""
    if current is None:
        raise gr.Error("Load a recording before re-indexing.")
    start = time.perf_counter()
    try:
        collection = reindex_collection(current, neural=neural)
    except AudioSleuthError as exc:
        raise gr.Error(str(exc)) from exc
    outputs = _collection_outputs(collection, time.perf_counter() - start)
    dispose(current)
    return outputs


def clear_callback(current: Collection | None) -> list[Any]:
    """Clear this session's index and files; Gradio's media cache expires separately."""
    dispose(current)
    return [
        None,
        "<div class='empty-banner'>Workspace cleared. Load a recording to begin.</div>",
        "",
        [],
        None,
        gr.update(choices=["All speakers"], value="All speakers"),
        None,
        "<div class='search-summary'>No active index.</div>",
        *_empty_results(),
        gr.update(choices=[ALL_EPISODES], value=ALL_EPISODES),
        gr.update(visible=False),
        gr.Tabs(selected="ingest"),
        gr.update(value=DEFAULT_ALPHA["offline"]),
        None,
    ]


def search_callback(
    query: str,
    alpha: float,
    speaker: str,
    confidence: float,
    minimum: float,
    collection: Collection | None,
    episode: str = ALL_EPISODES,
    scorer: str = "vectors",
    fusion: str = "linear",
) -> list[Any]:
    """Return actual clip files to Gradio Audio, not a full-source seek suggestion."""
    if collection is None:
        raise gr.Error("Load the demo or index an upload before searching.")
    start = time.perf_counter()
    try:
        hits = search(
            collection.index,
            query,
            alpha=alpha,
            speaker=None if speaker == "All speakers" else speaker,
            episode=None if episode == ALL_EPISODES else episode,
            min_confidence=confidence,
            min_score=minimum,
            top_k=MAX_RESULTS,
            text_scorer=scorer,  # type: ignore[arg-type]
            fusion=fusion,  # type: ignore[arg-type]
        )
    except AudioSleuthError as exc:
        raise gr.Error(str(exc)) from exc
    took = (time.perf_counter() - start) * 1000
    summary = (
        f"<div class='search-summary'><b>{len(hits)} moments found</b> in {took:.1f} ms"
        f" &nbsp; &middot; &nbsp; Text weight {alpha:.0%} / Audio weight {1 - alpha:.0%}"
        f"<br><small>Sorted by {FUSION_LABELS.get(fusion, fusion)}, "
        "not a calibrated confidence probability."
        " Clip players start at 0:00 within each exact source interval.</small></div>"
    )
    if not hits:
        summary += (
            "<p>No matches have usable query evidence or meet these filters. "
            "Try another query, speaker, weight, or threshold.</p>"
        )
    if alpha < 1 and collection.index.audio_backend != "CLAP" and not acoustic_query(query).any():
        summary += (
            "<p>Offline audio search found no supported tone words in this query. "
            "Audio similarity is zero; try loud, calm, excited, or use text weight 1.</p>"
        )
    names = {sha: name for name, sha in collection.episodes()}
    outputs: list[Any] = [summary]
    for position in range(MAX_RESULTS):
        if position < len(hits):
            hit = hits[position]
            outputs.extend(
                (
                    gr.update(visible=True),
                    hit_card(
                        hit,
                        query,
                        position + 1,
                        alpha=alpha,
                        scorer=scorer,
                        episode=names.get(hit.chunk.source_sha256, ""),
                    ),
                    hit.chunk.audio_path,
                )
            )
        else:
            outputs.extend((gr.update(visible=False), "", None))
    return outputs


def build_app() -> gr.Blocks:
    """Build two tabs with per-browser state and bounded model concurrency."""
    css = (ROOT / "static" / "style.css").read_text(encoding="utf-8")
    theme = gr.themes.Base(
        primary_hue="teal",
        neutral_hue="slate",
        font=["Inter", "Segoe UI", "Arial", "sans-serif"],
        font_mono=["Consolas", "monospace"],
    ).set(
        body_background_fill="#f3f6f7",
        body_background_fill_dark="#0b1a20",
        block_background_fill_dark="#13242c",
        block_border_color_dark="#24404b",
        input_background_fill_dark="#0f1d24",
    )
    with gr.Blocks(
        title="AudioSleuth | Find the moment. Hear the evidence.",
        theme=theme,
        css=css,
        analytics_enabled=False,
        delete_cache=(3600, 3600),
    ) as dashboard:
        state = gr.State(value=None, time_to_live=3600, delete_callback=dispose)
        gr.HTML("""
        <header class="hero">
          <div class="brand"><span class="brand-mark">a<span>s</span></span> AudioSleuth
            <span class="edition">RESEARCH STUDIO</span></div>
          <div class="eyebrow">MULTIMODAL AUDIO INTELLIGENCE</div>
          <h1>Find the moment.<br><em>Hear the evidence.</em></h1>
          <p>Search beyond the transcript. Follow the speaker, the tone, and the exact
          audio behind every answer.</p>
          <div class="hero-tags"><span>Speaker-aware</span><span>Dual-vector retrieval</span>
          <span>Source-linked playback</span>
          <span class="local">Local-first / offline demo</span></div>
          <div class="signal" aria-hidden="true">
          <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i>
          <i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
        </header>""")
        metrics = gr.HTML(
            "<div class='empty-banner'>Your listening workspace is ready. "
            "Load the bundled podcast to start instantly, without a GPU or account.</div>"
        )
        with gr.Tabs(selected="ingest") as tabs:
            with gr.Tab("1 · Ingest & index", id="ingest"):
                with gr.Row():
                    with gr.Column(scale=4, min_width=300, elem_classes=["as-card", "demo-card"]):
                        gr.HTML("""
                        <div class="eyebrow">NO SETUP REQUIRED</div>
                        <div class="panel-title">Market Signals demo</div>
                        <p class="panel-sub">A two-voice synthetic podcast about interest
                        rates, startup valuations, and clean energy, with bundled timestamps
                        and precomputed vectors. Loads instantly.</p>
                        <div class="demo-pills"><span>2 voices</span><span>13 moments</span>
                        <span>Offline</span></div>""")
                        demo = gr.Button(
                            "Load Pre-indexed Demo Audio", size="lg", variant="primary"
                        )
                        gr.HTML(
                            "<p class='fine'>Synthetic speech and scripted speaker labels. "
                            "Offline search uses TF-IDF and signal descriptors, "
                            "not neural emotion detection.</p>"
                        )
                    with gr.Column(scale=5, min_width=300, elem_classes=["as-card"]):
                        gr.HTML(
                            "<div class='panel-title'>Your own recording</div>"
                            "<p class='panel-sub'>WAV or MP3 · up to 15 minutes · 100 MiB · "
                            "needs the optional WhisperX install</p>"
                        )
                        upload = gr.File(
                            label="Podcast audio",
                            file_types=[".wav", ".mp3"],
                            type="filepath",
                            height=140,
                        )
                        neural = gr.Checkbox(
                            label="Neural embeddings (MiniLM + CLAP, optional ML install)",
                            value=False,
                        )
                        with gr.Row():
                            process = gr.Button("Process & Index", variant="primary")
                            add = gr.Button("Add to collection", variant="secondary")
                        gr.HTML(
                            "<p class='fine'><b>Process</b> replaces the collection; "
                            "<b>Add</b> indexes another episode beside the current ones, "
                            "keeping the collection's embedding mode. "
                            "Speaker labels need <code>HF_TOKEN</code>; without it speakers "
                            "stay UNKNOWN rather than guessed.</p>"
                        )
                with gr.Row(elem_classes=["workspace-row"]):
                    reindex = gr.Button(
                        "Re-index with current setting",
                        size="sm",
                        variant="secondary",
                        scale=0,
                        min_width=230,
                    )
                    clear = gr.Button(
                        "Clear workspace", size="sm", variant="stop", scale=0, min_width=160
                    )
                    gr.HTML(
                        "<p class='fine'>Re-index switches embeddings without re-transcribing. "
                        "Clear deletes this session's index and audio files.</p>"
                    )
                with gr.Accordion("Local backend readiness", open=False):
                    readiness = gr.Markdown(backend_status())
                    refresh = gr.Button("Refresh backend status", size="sm")
                    refresh.click(backend_status, outputs=[readiness], api_name="backend_status")
                with gr.Column(visible=False) as loaded:
                    status = gr.HTML("")
                    source = gr.Audio(
                        label="Source recording (latest episode)",
                        interactive=False,
                        type="filepath",
                    )
                    with gr.Accordion("Speaker timeline & provenance", open=False):
                        timeline = gr.Dataframe(
                            headers=[
                                "Episode",
                                "Speaker",
                                "Start",
                                "End",
                                "Align",
                                "Timing",
                                "Transcript",
                            ],
                            datatype=["str", "str", "str", "str", "number", "str", "str"],
                            column_widths=["15%", "11%", "9%", "9%", "6%", "8%", "42%"],
                            interactive=False,
                            wrap=True,
                            max_height=420,
                            label="Indexed speaker turns",
                        )
                        manifest = gr.File(
                            label="Provenance manifest (JSON)", interactive=False, height=60
                        )
            with gr.Tab("2 · Search & listen", id="search"):
                gr.HTML(
                    "<div class='panel-title'>Every result comes with receipts</div>"
                    "<p class='panel-sub'>Find what was said, how it sounded, "
                    "and who said it.</p>"
                )
                with gr.Row():
                    query = gr.Textbox(
                        label="Search the conversation",
                        scale=6,
                        placeholder="heated argument about interest rates",
                    )
                    submit = gr.Button("Find moments", variant="primary", scale=1)
                gr.Examples(
                    examples=[
                        ["heated argument about interest rates"],
                        ["speaker laughing about valuation"],
                        ["calm explanation of monetary policy"],
                        ["renewable energy and battery storage"],
                    ],
                    inputs=query,
                )
                with gr.Row():
                    alpha = gr.Slider(
                        0,
                        1,
                        value=DEFAULT_ALPHA["offline"],
                        step=0.05,
                        label="Semantic weight alpha",
                        info="0 = audio / tone only · 1 = transcript only",
                    )
                    speaker = gr.Dropdown(
                        choices=["All speakers"], value="All speakers", label="Speaker"
                    )
                    episode = gr.Dropdown(
                        choices=[ALL_EPISODES], value=ALL_EPISODES, label="Episode"
                    )
                with gr.Accordion("Ranking method & evidence filters", open=False):
                    with gr.Row():
                        confidence = gr.Slider(
                            0, 1, value=0, step=0.05, label="Minimum alignment quality"
                        )
                        minimum = gr.Slider(
                            -1,
                            1,
                            value=0,
                            step=0.05,
                            label="Minimum hybrid relevance",
                            info="Applies to linear fusion only",
                        )
                    with gr.Row():
                        scorer = gr.Radio(SCORERS, value="vectors", label="Text scorer")
                        fusion = gr.Radio(FUSIONS, value="linear", label="Score fusion")
                    gr.Markdown(
                        "Linear: `alpha * cosine(text) + (1-alpha) * cosine(audio)`. "
                        "Z-score standardizes each modality over the filtered candidates; "
                        "RRF sums `weight / (60 + rank)`. BM25 is unbounded, so combining it "
                        "with audio needs z-score or RRF.\n\n"
                        "Cosines are not probabilities. Offline tone mapping is a heuristic: "
                        "it cannot verify laughter, anger, identity, or psychological state."
                    )
                summary = gr.HTML(
                    "<div class='empty-banner'>Load an index in the first tab to begin.</div>"
                )
                result_outputs: list[Any] = []
                for position in range(MAX_RESULTS):
                    with gr.Group(visible=False, elem_classes=["result-card"]) as group:
                        card = gr.HTML("")
                        player = gr.Audio(
                            label=f"Match {position + 1} / exact audio excerpt",
                            interactive=False,
                            type="filepath",
                            waveform_options=gr.WaveformOptions(
                                waveform_color="#8296a7",
                                waveform_progress_color="#16a394",
                            ),
                        )
                    result_outputs.extend((group, card, player))
        gr.HTML("""
        <footer>AudioSleuth &nbsp; / &nbsp; Evidence before inference.
        <span>Local research demo. Synthetic fixtures are labeled. No analytics.</span></footer>""")
        index_outputs = [
            state,
            metrics,
            status,
            timeline,
            source,
            speaker,
            manifest,
            summary,
            *result_outputs,
            episode,
            loaded,
            tabs,
            alpha,
        ]
        demo.click(
            load_demo_callback,
            inputs=[state],
            outputs=index_outputs,
            api_name="load_demo",
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
        process.click(
            process_callback,
            inputs=[upload, neural, state],
            outputs=index_outputs,
            api_name="process_audio",
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
        add.click(
            add_callback,
            inputs=[upload, neural, state],
            outputs=index_outputs,
            api_name="add_audio",
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
        reindex.click(
            reindex_callback,
            inputs=[neural, state],
            outputs=index_outputs,
            api_name="reindex",
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
        clear.click(
            clear_callback,
            inputs=[state],
            outputs=index_outputs + [upload],
            api_name="clear_workspace",
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
        search_inputs = [
            query,
            alpha,
            speaker,
            confidence,
            minimum,
            state,
            episode,
            scorer,
            fusion,
        ]
        submit.click(
            search_callback,
            inputs=search_inputs,
            outputs=[summary, *result_outputs],
            api_name="search",
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
        query.submit(
            search_callback,
            inputs=search_inputs,
            outputs=[summary, *result_outputs],
            api_name=False,
            concurrency_limit=1,
            concurrency_id="pipeline",
        )
    return dashboard


def main() -> None:
    """Bind only loopback by default; never create a public sharing tunnel."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    build_app().queue(max_size=16, default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        server_port=args.port,
        share=False,
        inbrowser=False,
        max_file_size="100mb",
        show_error=False,
    )


if __name__ == "__main__":
    main()
