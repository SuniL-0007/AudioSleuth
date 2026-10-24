"""Score retrieval configurations against time-span qrels and print Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from evaluation.metrics import Span, score_query
from retrieval import search
from schemas import AudioSleuthError
from service import Collection, create_collection, dispose, reindex_collection

QRELS = Path(__file__).with_name("qrels_demo.json")
CONFIGS: list[dict[str, Any]] = [
    {"label": "text only", "alpha": 1.0, "text_scorer": "vectors", "fusion": "linear"},
    {"label": "BM25 only", "alpha": 1.0, "text_scorer": "bm25", "fusion": "linear"},
    {"label": "audio only", "alpha": 0.0, "text_scorer": "vectors", "fusion": "linear"},
    {"label": "linear", "alpha": 0.35, "text_scorer": "vectors", "fusion": "linear"},
    {"label": "linear", "alpha": 0.65, "text_scorer": "vectors", "fusion": "linear"},
    {"label": "z-score", "alpha": 0.65, "text_scorer": "vectors", "fusion": "zscore"},
    {"label": "RRF", "alpha": 0.65, "text_scorer": "vectors", "fusion": "rrf"},
    {"label": "BM25 + z-score", "alpha": 0.65, "text_scorer": "bm25", "fusion": "zscore"},
    {"label": "BM25 + RRF", "alpha": 0.65, "text_scorer": "bm25", "fusion": "rrf"},
]
DEPTH = 10  # Ranks retrieved per query; metrics read at most this far.


def load_qrels(path: Path) -> list[dict[str, Any]]:
    """Parse queries and validate spans so typos fail before any ranking runs."""
    queries = json.loads(path.read_text(encoding="utf-8"))["queries"]
    for query in queries:
        query["spans"] = [Span(**span) for span in query["spans"]]
        if not query["spans"]:
            raise ValueError(f"Query {query['id']} has no relevant spans.")
    return list(queries)


def evaluate(
    collection: Collection, queries: list[dict[str, Any]], config: dict[str, Any], *, k: int = 5
) -> list[dict[str, Any]]:
    """Per-query metrics for one configuration of retrieval.search options."""
    options = {key: value for key, value in config.items() if key != "label"}
    rows = []
    for query in queries:
        hits = search(collection.index, query["query"], top_k=DEPTH, **options)
        metrics = score_query([hit.chunk for hit in hits], query["spans"], k=k)
        rows.append({"id": query["id"], "type": query["type"], **metrics})
    return rows


def mean_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    names = [key for key in rows[0] if key not in ("id", "type")]
    return {name: fmean(row[name] for row in rows) for name in names}


def markdown_table(header: list[str], body: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def run(
    *, qrels: Path = QRELS, audio: Path | None = None, neural: bool = False, k: int = 5
) -> list[dict[str, Any]]:
    """Evaluate every backend/alpha pair; each result carries per-query rows.

    Without audio, the bundled demo is used; an upload is transcribed once and
    re-indexed for the neural pass so both backends rank identical chunks.
    """
    queries = load_qrels(qrels)
    results = []
    base = create_collection(upload=audio)
    collections = [base]
    try:
        if neural:
            collections.append(reindex_collection(base, neural=True))
            index = collections[-1].index
            if index.text_backend != "sentence-transformers" and index.audio_backend != "CLAP":
                raise AudioSleuthError(
                    "--neural was requested, but MiniLM and CLAP both fell back to the offline "
                    "encoders, so the neural rows would only repeat the offline ones. Install "
                    "requirements-ml.txt (or sentence-transformers and transformers) first."
                )
        for collection in collections:
            backend = f"{collection.index.text_backend} + {collection.index.audio_backend}"
            for config in CONFIGS:
                rows = evaluate(collection, queries, config, k=k)
                results.append({"backend": backend, **config, "queries": rows})
    finally:
        for collection in collections:
            dispose(collection)
    return results


def report(results: list[dict[str, Any]], *, per_query: bool = False) -> str:
    """Overall means, then means per query type, then optional per-query rows."""
    names = list(mean_metrics(results[0]["queries"]))
    sections = []
    groups = ["all"] + sorted({row["type"] for row in results[0]["queries"]})
    for group in groups:
        body = []
        for result in results:
            rows = [r for r in result["queries"] if group == "all" or r["type"] == group]
            means = mean_metrics(rows)
            body.append(
                [result["backend"], result["label"], f"{result['alpha']:.2f}"]
                + [f"{means[name]:.3f}" for name in names]
            )
        title = f"### {group} queries (n={len(rows)})"
        header = ["backend", "config", "alpha", *names]
        sections.append(title + "\n\n" + markdown_table(header, body))
    if per_query:
        for result in results:
            body = [
                [row["id"], row["type"]] + [f"{row[name]:.3f}" for name in names]
                for row in result["queries"]
            ]
            title = (
                f"### per query: {result['backend']}, {result['label']}, "
                f"alpha={result['alpha']:.2f}"
            )
            sections.append(title + "\n\n" + markdown_table(["id", "type", *names], body))
    return "\n\n".join(sections)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=QRELS)
    parser.add_argument("--audio", type=Path, help="recording judged by --qrels (default: demo)")
    parser.add_argument("--neural", action="store_true", help="also re-index with MiniLM + CLAP")
    parser.add_argument("--k", type=int, default=5, help="cutoff for P@k, R@k, nDCG@k")
    parser.add_argument("--per-query", action="store_true")
    parser.add_argument("--json", type=Path, help="write full results to this file")
    args = parser.parse_args()
    if not 1 <= args.k <= DEPTH:
        parser.error(f"--k must be between 1 and {DEPTH}.")
    try:
        results = run(qrels=args.qrels, audio=args.audio, neural=args.neural, k=args.k)
    except AudioSleuthError as exc:
        parser.exit(1, f"error: {exc}\n")
    print(report(results, per_query=args.per_query))
    if args.json:
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
