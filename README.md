# AudioSleuth

**Zero-Shot Multi-Speaker Podcast Retrieval & Audio Provenance System**

A complete local demonstration with original, bundled two-voice speech,
speaker-aware chunking, a NumPy dual-vector index, hybrid retrieval, and
waveform playback of exact source slices.

**Start here:** launch [app.py](app.py), click **Load Pre-indexed Demo Audio**,
then open **Search & listen**. No GPU, token, model download, FFmpeg, or online
speech service is required for this path.

## What is implemented, and what is actually being demonstrated?

| Stage | Optional neural path | Always-available offline demonstration |
|---|---|---|
| ASR and alignment | Local WhisperX ASR and its language-specific forced aligner | Bundled original speech with TTS word-boundary events |
| Diarization | WhisperX's pyannote pipeline, with an authorized Hugging Face token | Two scripted, explicitly labeled speaker tracks |
| Dynamic chunks | Speaker changes, silence, punctuation, lexical topic shifts, 120-second cap | The same chunking code |
| Text vectors | `sentence-transformers/all-MiniLM-L6-v2`, on CPU | Word/bigram TF-IDF vectors |
| Audio vectors | LAION-CLAP weights via Transformers, 48 kHz windows pooled across the entire chunk | Nine measured signal descriptors |
| Query-to-audio comparison | CLAP's text encoder in the shared text/audio space | Explicit tone-word-to-descriptor mapping |
| Index | Session-local, normalized NumPy vector matrices | Same; bundled precomputed vectors load immediately |
| Provenance player | Timestamped source WAV excerpts with waveform controls | Same |

**The offline mode is not zero-shot neural semantic search or an emotion
classifier.** Its TF-IDF and acoustic descriptor backends are clearly shown
in the UI. For example, "laughing" is a query cue for brightness/dynamics in
the fallback, not an acoustic verification that someone laughed. The demo
contains synthesized "ha ha" speech, not recorded natural human laughter.
CLAP is also a general audio/text model, not a calibrated psychological
state detector.

The optional model adapters are implemented against the upstream APIs linked
below. Real MiniLM and LAION-CLAP inference runs on transformers 4.57
(Python 3.12) and 5.20 (Python 3.14). WhisperX CPU transcription has been
verified on Python 3.12, including a 12-minute real interview (see
**Retrieval evaluation**). Speaker diarization (pyannote
`speaker-diarization-community-1`, CPU) has been verified on that interview:

- Every chunk's label matched the actual speaker. The host's questions became
  `SPEAKER_00` and the guest's answers `SPEAKER_01`.
- 99.9% of words overlapped a diarized interval by at least half their
  duration.
- Speaker-aware chunking split three previously mixed host/guest chunks at the
  handover, giving 59 chunks instead of 57.

Diarization needs an `HF_TOKEN` with access to the gated model; without one,
speakers stay `UNKNOWN`. The lightweight `.venv` remains usable independently.

This is a **working, production-oriented local demonstration**, not a claim
of audited production readiness, benchmarked recognition accuracy, or a
multi-tenant hosted service. Deployment requirements are listed below.

## Quick start

### Windows PowerShell

From this project directory:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open **http://127.0.0.1:7860**.

The delivered environment already exists, so on this machine:

```powershell
cd <your clone of AudioSleuth>
.\.venv\Scripts\python.exe app.py
```

If the app is already running, open its URL instead of starting a second copy.
Use `python app.py --port 7861` to intentionally run another instance.
Stop a foreground server with Ctrl+C.

The prepared neural environment can run the same dashboard with the actual
ML backends installed:

```powershell
.\.venv-ml\Scripts\python.exe app.py
```

Tick **Neural embeddings** and click **Re-index with current setting** to
test real MiniLM + CLAP on the bundled recording. The initial
**Load Pre-indexed Demo Audio** button intentionally stays instantaneous and
offline in either environment.

### Linux / macOS / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python app.py
```

The source targets Python **3.10+**; pip selects compatible dependency releases.
Python **3.11** is recommended for optional speech-model compatibility.
The offline application was tested on Windows with Python 3.11.15 and 3.14.5
and Gradio 5.50.0. CI (`.github/workflows/ci.yml`) runs lint, type checks,
tests, and the retrieval evaluation on Ubuntu and Windows with Python 3.11
through 3.14. WhisperX (uploads) requires Python **3.10–3.13**; the neural
embedding stack alone also runs on 3.14. Gradio is constrained to 5.x because 6.x changes
several UI APIs.

The first dependency installation needs internet access. The bundled demo
needs no network afterward. The app disables Gradio analytics, uses system
fonts, binds only to loopback, and never creates a public Gradio sharing tunnel.

## Presentation walkthrough

1. In **Ingest & index**, click **Load Pre-indexed Demo Audio** in the demo card.
2. The app opens **Search & listen** automatically. The summary row shows the
   episode, two speaker labels, 13 retrievable moments, and the load time
   (about 0.1–0.2 seconds here, depending on hardware and cache state).
3. Search **heated argument about interest rates**.
4. The top example result is `SPEAKER_01`, from **00:15.545 to 00:26.420**.
   Its actual playable excerpt is **10.875 seconds**, not the whole podcast.
5. Move `alpha` toward 1 for transcript relevance or toward 0 for audio cues.
   Click **Find moments** to apply updated weights and filters. The slider
   resets to 0.65 for offline indexes and 0.9 for neural ones when an index loads.
6. Filter `SPEAKER_00` to retrieve only that speaker's chunks. Under
   **Ranking method & evidence filters**, switch the text scorer to BM25 or the
   fusion to z-score / RRF.
7. Try:
   - `speaker laughing about valuation`
   - `calm explanation of monetary policy`
   - `renewable energy and battery storage`
8. Back in **Ingest & index**, open **Speaker timeline & provenance** to see
   every indexed turn and download the provenance manifest.

Use **Local backend readiness** to inspect installed prerequisites without
downloading weights or displaying tokens. **Re-index with current setting**
uses the embedding checkbox and retains the existing words and speaker labels;
this allows the bundled recording to exercise real neural embeddings without
rerunning ASR. **Clear workspace** removes the active index and its owned
source/snippet files, resets results, clears the upload field, and returns to
the ingest tab. **Add to collection** keeps the collection's current embedding
mode, so adding an episode never silently turns a neural index offline. Gradio's
separate media cache remains subject to its expiry policy.

Up to 10 results display speaker (color-coded per label), transcript with escaped query highlights, start/end
timestamps, combined relevance, component cosines, alignment quality,
speaker-overlap evidence, timing method, and source SHA-256.

The recording is **126.055 seconds** of original scripted synthetic speech,
with **277 timed words**, generated locally using two Windows system voices.
It is not a copyrighted podcast excerpt, a voice clone, or a real interview.
Runtime playback works on any supported platform; Windows is needed only to
regenerate these particular bundled TTS assets.

## Architecture and file layout

```text
AudioSleuth/
|-- app.py                    # Gradio dashboard, callbacks, safe highlights
|-- ingestion.py              # WAV/MP3 decoding, WhisperX, speaker assignment
|-- chunking.py               # Dynamic turns and exact audio snippets
|-- indexing.py               # Neural/fallback encoders and NumPy dual index
|-- bm25.py                   # Sparse Okapi BM25 text scorer
|-- retrieval.py              # Linear / z-score / RRF fusion and metadata filters
|-- service.py                # Atomic multi-episode collection lifecycle
|-- schemas.py                # Typed words, intervals, chunks, search hits
|-- requirements.txt          # Lightweight runtime
|-- requirements-ml.txt       # Optional neural stack
|-- requirements-dev.txt      # Tests and static checks
|-- pyproject.toml
|-- static/
|   `-- style.css
|-- assets/
|   |-- script.json           # Original script and voice settings
|   |-- demo.wav              # Bundled playable two-speaker audio
|   |-- demo.json             # Word timings, intervals, source hash
|   |-- demo_index.json       # Payloads, vocabulary, index version
|   `-- demo_index.npz        # Precomputed vectors and IDF; no pickle
|-- tools/
|   |-- synthesize_demo.ps1   # Optional offline Windows TTS regeneration
|   |-- build_demo.py         # Audio/manifest/index assembly
|   |-- check_backends.py     # Real-model smoke tests, never fallback successes
|   `-- fetch_eval_audio.py   # Download + verify the real evaluation excerpt
|-- evaluation/
|   |-- metrics.py            # P@k, R@k, MRR, nDCG@k, MAP over time-span qrels
|   |-- qrels_demo.json       # 20 judged queries: lexical, paraphrase, tone
|   |-- qrels_nasa.json       # 16 judged queries on a real NASA interview excerpt
|   `-- run_eval.py           # Config-grid evaluation CLI
`-- tests/
    |-- conftest.py
    |-- test_ingestion.py
    |-- test_chunking.py
    |-- test_retrieval.py
    |-- test_fusion.py
    |-- test_collection.py
    |-- test_metrics.py
    |-- test_app.py
    `-- test_recheck.py
```

### Data flow

```text
WAV / MP3
   |
bounded decode -> 16 kHz mono -> original-source SHA-256
   |
WhisperX ASR -> forced alignment -> words with timing provenance
   |                                     |
pyannote speaker intervals --------------> maximum temporal-overlap assignment
                                         |
                             speaker-stratified dynamic chunks
                                         |
                +------------------------+------------------------+
                |                                                 |
        transcript embeddings                               audio embeddings
                |                                                 |
                +---------------- NumPy dual index ---------------+
                                         |
                             metadata filters + hybrid cosine
                                         |
                              ranked timestamped excerpts
                                         |
                             waveform player + JSON provenance
```

The demo enters the same pipeline after ASR/diarization, using verified
bundled fixtures, and loads precomputed text/audio vectors.

## Pipeline details

### 1. Audio loading and word-level speaker assignment

- WAV and MP3 are decoded with `soundfile`; FFmpeg is a fallback for codecs
  unsupported by the installed libsndfile.
- Downmix channels and polyphase-resample to 16 kHz mono.
- Reject files over 100 MiB, recordings over 15 minutes, empty/very short audio,
  nonfinite samples, and unsupported file extensions.
- CPU WhisperX uses `int8`; CUDA uses `float16`. Model name defaults to `small`
  and can be changed with `AUDIOSLEUTH_ASR_MODEL`.
- Forced alignment provides word timestamps. Tokens that cannot be aligned
  are explicitly interpolated where possible, marked **estimated**, and
  assigned alignment quality zero. Impossible timings produce an error.

For word interval \(W=[s_w,e_w)\), choose the speaker with greatest overlap:

\[
\hat{p}(w)=\arg\max_p
\left|W\cap\bigcup_{i:\,p_i=p}[s_i,e_i)\right|.
\]

Unioning intervals prevents duplicate tracks from inflating overlap.
Ties are resolved deterministically by speaker label. Nearest-turn assignment
is allowed only within 250 ms and has **zero overlap confidence**.
Otherwise the word is `UNKNOWN`. Simultaneous speech is reduced to one label;
this system does not separate overlapping voices or guarantee that a snippet
contains only one audible speaker.

### 2. Speaker-stratified dynamic chunking

- Contiguous same-speaker words form turns.
- Speaker changes and silences longer than 1.2 seconds start new turns.
- After at least eight seconds, punctuation with a 350 ms pause or a strong
  lexical-cohesion change can start a new chunk.
- The topic-shift test uses adjacent transcript windows and TF-IDF similarity.
  It is a transparent lexical heuristic, not an unvalidated neural semantic
  change-point detector.
- Long monologues split preferentially at punctuation, otherwise at a word
  boundary. A hard cap of **120,000 milliseconds** is enforced.
- No fixed token-count slicing is used. All words are preserved in order.

### 3. Dual-vector indexing

**Neural text:** sentence-transformers MiniLM produces normalized vectors.
Long transcripts are split at the tokenizer's actual context capacity.
Every token window is embedded and token-count-weighted before normalization,
so the end of a 120-second monologue is not silently discarded by MiniLM.

**Neural audio:** Transformers loads `laion/clap-htsat-unfused`, the actual
LAION model weights. CLAP consumes 48 kHz waveform windows. Every window
of a long chunk is embedded; normalized window vectors are mean-pooled and
renormalized. This avoids silently indexing only the first ten seconds.
The query is encoded with the **same CLAP model's text encoder**.

**Offline text:** normalized unigram/bigram TF-IDF with a bounded vocabulary.
This is lexical matching; it cannot promise synonym or paraphrase retrieval.

**Offline audio:** measured loudness, quietness, brightness, darkness,
modulation/dynamics, steadiness, noisiness, tonality, and silence. Query
tone words map explicitly to those axes. If there are no supported tone
words, the acoustic query vector is zero and the UI explains this.

NumPy matrices are an exact cosine index, not approximate nearest-neighbor
search. This is appropriate for bounded, single-recording collections.
The index stores row-aligned payloads and validates the bundled source hash,
chunk order, text, boundaries, and dimensions before loading precomputed vectors.
No untrusted pickle is loaded.

Payload example:

```json
{
  "chunk_id": "bf94c59f951d-0002",
  "speaker_id": "SPEAKER_01",
  "start_ms": 15545,
  "end_ms": 26420,
  "start_sec": 15.545,
  "end_sec": 26.42,
  "transcript": "I completely disagree about interest rates! ...",
  "confidence": 1.0,
  "speaker_confidence": 1.0,
  "timing": "synthetic",
  "audio_path": "bf94c59f951d-0002.wav",
  "source_sha256": "the full 64-character hash is included in the actual manifest"
}
```

The example above abbreviates only the documentation display; actual payloads
contain the full transcript and full hash. Runtime payloads point to real
clip files. The exported manifest uses basenames rather than leaking server
filesystem paths.

### 4. Retrieval and score meaning

All vectors are L2-normalized:

\[
\mathrm{score}(q,c)=
\alpha\,(t_q^\top t_c)+(1-\alpha)\,(a_q^\top a_c).
\]

- `alpha=1`: transcript-only ranking.
- `alpha=0`: acoustic-only ranking.
- A zero-weight modality is not invoked. Text-only search does not require a
  working acoustic encoder, and audio-only search does not require MiniLM.
- Queries with no usable evidence in either selected modality return no
  matches instead of arbitrary all-zero ranked chunks.
- Speaker and minimum **alignment quality** filters apply before top-k.
- Minimum **hybrid relevance** is a separate filter.
- Cosines can be negative; the implementation does not pretend that they
  are probabilities or silently remap them to percentages.
- Query embeddings never silently switch models after indexing. If the
  original encoder becomes unavailable, the app asks for re-indexing.
- Ties are stable by source start time.

**Text scorer.** Besides the index's text vectors (TF-IDF or MiniLM), every
index carries an Okapi BM25 scorer (`k1=1.5`, `b=0.75`, English stopwords
removed), built from the chunk transcripts in milliseconds.

**Fusion.** The formula above is the default `linear` fusion. Two alternatives
avoid mixing incomparable score scales:

- `zscore`: each modality is standardized over the filtered candidates, then
  weighted by `alpha`.
- `rrf`: weighted reciprocal rank fusion,
  `alpha/(60+rank_text) + (1-alpha)/(60+rank_audio)`.

In both, a modality with no query evidence contributes nothing, so it cannot
reorder the other modality's ranking. Evidence is judged among the filtered
candidates, so a term that occurs only in another episode returns no matches
rather than zero-score chunks. BM25 scores are unbounded, so BM25 with
`0<alpha<1` requires `zscore` or `rrf`. Minimum hybrid relevance applies only
to `linear`.

**Episodes.** **Add to collection** indexes another recording next to the
current ones in one shared index, so IDF statistics and normalization cover
the whole corpus. The **Episode** filter (by source SHA-256) applies before
ranking, like the speaker filter. Duplicate recordings are rejected, and the
provenance manifest lists every source.

### 5. What "exact millisecond provenance" means

At 16 kHz, one millisecond contains exactly 16 samples. The snippet saved for
`[start_ms, end_ms)` contains:

```python
recording.samples[start_ms * 16 : end_ms * 16]
```

Start times round outward down; end times round outward up, bounded to the
last full millisecond of available audio. Snippets use float WAV so tests
can verify **sample-for-sample equality** against the normalized source.
The UI plays this actual excerpt rather than seeking approximately into
the whole source.

This guarantees precise indexing/playback coordinates, **not** millisecond
accuracy of recognition or forced alignment. Demo word starts come from
TTS boundary events; word ends use the next event and include inter-word
pauses. MP3 decoding/encoder delay and resampling conventions also affect
the relationship to another player's timeline.

A SHA-256 identifies source bytes and detects accidental fixture mismatch.
It is not a digital signature, authenticity certificate, watermark, or
proof that a recording has never been edited.

## Optional real-audio / neural setup

Use a separate environment if you need to preserve a lightweight presentation
installation. Heavy ML dependencies have tighter Python/platform constraints.

```bash
python3.11 -m venv .venv-ml
source .venv-ml/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ml.txt
```

The optional stack uses **Transformers CLAP**, not the legacy `laion-clap`
Python package. This resolves a real dependency conflict: WhisperX 3.8.6
requires NumPy 2.1+, whereas `laion-clap` 1.1.7 requires NumPy below 2.
Older WhisperX 3.7 also requires NumPy 2. Do not install the legacy CLAP
package into this environment. Transformers is constrained to its compatible
4.x API.

Install FFmpeg for formats that libsndfile cannot decode and for optional
file-based decoder integrations, using your platform package manager:

```bash
# Debian / Ubuntu / WSL
sudo apt-get update
sudo apt-get install ffmpeg

# macOS with Homebrew
brew install ffmpeg
```

On Windows, install FFmpeg and ensure `ffmpeg` is on PATH. For CUDA, install
the PyTorch/CUDA stack recommended by WhisperX and PyTorch for your platform;
the app does not assume that a CUDA runtime is present.

For diarization:

1. Accept the model's conditions at
   [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1).
2. Create a **Read** token at <https://huggingface.co/settings/tokens>.
3. Store it as an environment variable. Never commit it, paste it into a chat,
   or write it into a project file.

```bash
# Linux / macOS: add to ~/.bashrc or ~/.zshrc to persist, then open a new shell
export HF_TOKEN="your-read-token"
python app.py
```

```powershell
# Windows: persists for new terminals (run in your own terminal, then reopen it)
setx HF_TOKEN "your-read-token"
# Or only for the current session:
$env:HF_TOKEN = "your-read-token"
python app.py
```

Upload a WAV/MP3 and click **Process & Index**. Check the neural-embedding
checkbox if you want MiniLM and CLAP instead of the offline encoders.
Initial model downloads and transcription can take minutes.

To test the actual adapters rather than the fallback path:

```bash
python tools/check_backends.py --backend text
python tools/check_backends.py --backend audio
python tools/check_backends.py --backend asr --asr-model tiny.en
```

These commands may download weights and return nonzero on failure. The ASR
check transcribes a short excerpt of the original synthetic speech; it does
not read the bundled transcript as a substitute. Authenticated diarization
still needs your token and acceptance of the model's access conditions.

All actual transcription/embedding inference happens locally. Model
downloads contact the respective model hosts; the program does not send
your uploaded audio to a hosted transcription API.

### Explicit fallback behavior

| Condition | Behavior |
|---|---|
| No CUDA | WhisperX attempts local CPU int8 inference |
| WhisperX absent or alignment fails | Visible error; upload is not indexed; demo button remains available |
| Missing token / diarization failure | Real transcript retained; speaker labels are UNKNOWN with a notice |
| MiniLM unavailable | Visible TF-IDF fallback, with matching lexical query encoder |
| CLAP unavailable | Visible signal-descriptor fallback, with matching tone-query mapper |
| No recognized offline tone words | Audio similarity is zero; UI explains how to change the query/weight |
| Invalid audio or input | Actionable error, no fabricated output |
| Failed replacement upload | Existing collection remains available |
| Invalid/mismatched bundled assets | Explicit integrity error, not an empty success |

The app deliberately does **not** attach the sample podcast transcript to
an unrelated uploaded file. Such a fallback would fabricate provenance.

## Retrieval evaluation

```bash
python -m evaluation.run_eval              # offline backends
python -m evaluation.run_eval --per-query  # add per-query tables
python -m evaluation.run_eval --neural     # also MiniLM + CLAP (needs .venv-ml)
# Multi-episode: demo + your recording(s), judged by merged qrels files
python -m evaluation.run_eval --audio ep2.wav --qrels evaluation/qrels_demo.json --qrels ep2.json
python -m evaluation.run_eval --no-demo --audio ep2.wav --qrels ep2.json
```

Relevance judgments are **time spans** (seconds, graded 1 or 2), not chunk
IDs, so they stay valid when the chunking changes. A retrieved chunk is
relevant when it overlaps a judged span by at least half of the shorter
interval. Each span is credited once, so duplicate hits cannot inflate
scores. `evaluation/qrels_demo.json` holds 20 queries in three groups:
lexical (shared words), paraphrase (no shared content words), and tone.

Every recording is indexed into one shared collection. Spans are tied to an
episode by a SHA-256 prefix: a file-level `"source_sha256_prefix"` applies to
all of its spans, and a span's own `"source"` overrides it. A span from one
episode therefore cannot be credited by the same time range in another. If a
qrels file names an episode that was not indexed, the run stops instead of
silently scoring those queries as zero. A recording's hash is shown in the
app's Episode filter and in its provenance manifest.

Offline results on the bundled demo (13 chunks, k=5):

| Config | alpha | MRR all | MRR lexical | MRR paraphrase | MRR tone | nDCG@5 all |
|---|---|---|---|---|---|---|
| TF-IDF only | 1.00 | 0.630 | 1.000 | 0.228 | 0.667 | 0.562 |
| BM25 only | 1.00 | 0.600 | 1.000 | 0.143 | 0.667 | 0.531 |
| Audio descriptors only | 0.00 | 0.202 | 0.000 | 0.000 | 0.672 | 0.181 |
| TF-IDF + audio, linear | 0.65 | 0.690 | 1.000 | 0.228 | 0.867 | 0.633 |
| TF-IDF + audio, RRF | 0.65 | 0.690 | 1.000 | 0.228 | 0.867 | 0.622 |
| BM25 + audio, z-score | 0.65 | 0.660 | 1.000 | 0.143 | 0.867 | 0.602 |

How to read the table:
- Lexical methods solve lexical queries and fail on paraphrases. That gap is
  what neural text embeddings (`--neural`) should close.
- The audio descriptors help only on tone queries.
- On a 13-chunk corpus, the fusion methods barely differ. A larger
  multi-episode corpus is needed to separate them.
- Tone results flatter the descriptors, because the demo's two synthetic
  voices also differ in gain.

Neural results (`--neural`: MiniLM + CLAP, CPU, transformers 5.20). MRR:

| Config | alpha | all | lexical | paraphrase | tone | nDCG@5 all |
|---|---|---|---|---|---|---|
| MiniLM only | 1.00 | **0.854** | 1.000 | **1.000** | 0.514 | **0.818** |
| CLAP audio only | 0.00 | 0.352 | 0.371 | 0.417 | 0.254 | 0.238 |
| MiniLM + CLAP, linear | 0.65 | 0.782 | 0.929 | 0.833 | 0.552 | 0.772 |
| MiniLM + CLAP, z-score | 0.65 | 0.786 | 0.905 | 0.893 | 0.524 | 0.785 |
| MiniLM + CLAP, RRF | 0.65 | 0.624 | 0.738 | 0.726 | 0.371 | 0.670 |
| BM25 + CLAP, z-score | 0.65 | 0.792 | 1.000 | 0.535 | **0.850** | 0.664 |

What the neural run shows:
- MiniLM closes the paraphrase gap: TF-IDF's paraphrase MRR goes from 0.228
  to 1.000. All seven paraphrase queries rank a correct moment first,
  including five that TF-IDF missed or ranked low.
- CLAP adds little here. It is weaker on tone queries than the
  nine hand-made descriptors (0.254 vs 0.672 MRR), and mixing it in lowers
  MiniLM's overall score. CLAP is a general audio-event model and was not
  trained on speech prosody. On this synthetic speech it acts as noise more
  than as a tone signal.
- The evaluation set is small (20 queries, 13 chunks), so read these as
  directions, not firm numbers.

### Real-recording evaluation (demo + NASA interview)

The second corpus is real two-person speech: a 12-minute excerpt of NASA's
*Houston We Have a Podcast*, episode 424 "AI at NASA". Credit: NASA Johnson
Space Center; NASA media is used here for education, with no endorsement
implied. The audio is never committed. `tools/fetch_eval_audio.py` downloads
the episode, checks its SHA-256, and writes the excerpt to the git-ignored
`evaluation/data/`. `evaluation/qrels_nasa.json` holds 16 judged queries:
8 lexical and 8 paraphrase. It has no tone queries, because the interview is
uniformly calm. The judgments were drafted with AI assistance from a WhisperX
transcript and should be reviewed by a person before being reported.

```bash
python tools/fetch_eval_audio.py   # once; ~25 MB download
python -m evaluation.run_eval --neural --audio evaluation/data/nasa_ai_at_nasa.wav \
  --qrels evaluation/qrels_demo.json --qrels evaluation/qrels_nasa.json
```

The whole run uses one shared index: 70 chunks, of which WhisperX `small` on
CPU produced 57 from the excerpt, and 36 queries. MRR on Python 3.12:

| Config | alpha | demo lexical | demo paraphrase | demo tone | NASA lexical | NASA paraphrase | all |
|---|---|---|---|---|---|---|---|
| TF-IDF only | 1.00 | 1.000 | 0.048 | 0.667 | 1.000 | 0.448 | 0.637 |
| BM25 only | 1.00 | 1.000 | 0.143 | 0.667 | 1.000 | 0.450 | 0.656 |
| MiniLM only | 1.00 | 1.000 | **0.690** | 0.298 | 1.000 | **0.567** | **0.726** |
| MiniLM + CLAP, linear | 0.65 | 0.929 | 0.687 | 0.348 | 1.000 | 0.479 | 0.701 |
| MiniLM + CLAP, RRF | 0.65 | 0.569 | 0.548 | 0.117 | 0.768 | 0.271 | 0.468 |
| BM25 + CLAP, z-score | 0.65 | 1.000 | 0.250 | **0.750** | 1.000 | 0.451 | 0.691 |

What the larger, real corpus adds:
- **Real paraphrases are harder.** On the NASA episode, MiniLM raises
  paraphrase MRR from 0.448 (TF-IDF) to 0.567. That is a clear gain, but far
  from the perfect score it reached on the scripted demo alone.
- **Distractors separate the methods.** With 57 more chunks in the index,
  TF-IDF's demo-paraphrase MRR falls from 0.228 to 0.048, while MiniLM holds
  0.690.
- **RRF is a poor fit for CLAP.** CLAP produces a full ranking for every
  query, so rank fusion gives that weak signal full weight. Score-level fusion
  (linear or z-score) degrades far less.
- **Lexical queries are solved** by every text method on both episodes.
- **The neural default α=0.9 holds up.** Sweeping α with linear fusion on this
  corpus gives MRR 0.701 at 0.65, 0.719 at 0.8, **0.729 at 0.9**, 0.724 at 0.95,
  and 0.726 at 1.0. The app therefore resets the slider to 0.9 for neural
  indexes and keeps 0.65 offline, where MRR is flat across α.

## Validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m mypy
python -m ruff check .
python -m ruff format --check .
```

The delivered test suite exercises:

- Stereo resampling and WAV/MP3 loading.
- Maximum speaker overlap, deterministic ties, duplicated intervals, and unknown speakers.
- Invalid/nonfinite timestamps and visibly estimated word times.
- Speaker changes, silences, long-monologue caps, and preservation of every word.
- Exact equality between all demo snippet samples and their source slices.
- Actual demo retrieval for interest rates, valuation, and battery storage.
- The hybrid-score formula at alpha endpoints and intermediate weights.
- Speaker/confidence filtering, empty results, and invalid queries.
- Explicit missing-model fallback without fabricated transcript data.
- Whole-chunk CLAP window coverage with a controlled adapter fixture.
- Pickle-free precomputed-index round trips and provenance tamper detection.
- Dashboard creation, escaped highlights, independent collections, and scoped cleanup.
- Missing word scores, invalid aligner timestamps, and partial timing anchors.
- Zero-weight encoder independence and unsupported-query empty results.
- Full-token neural text pooling and the Transformers CLAP adapter contract.
- Corrupt IDF rejection, re-indexing, and explicit workspace clearing.
- Hand-computed BM25 scores, RRF and z-score semantics, and the episode filter.
- Multi-episode collections: shared index, duplicate rejection, and manifests.
- Hand-computed IR metrics, span judging, and an end-to-end demo evaluation.

- Upstream API changes (AttributeError) degrading to visible offline fallbacks.
- Episode-aware qrels, merged qrels files, and unindexed-episode detection.

Current result: **82 tests passed** (also on Python 3.12 with the pinned ML stack). Mypy (ten modules), Ruff lint, and Ruff
format are clean on Windows with Python 3.14.5.

Browser verification (Playwright driving Edge) loaded the demo, checked the
automatic switch to **Search & listen**, searched, and captured light and dark
themes at 1440 px and 390 px. No width showed horizontal overflow. Earlier
HTTP-client checks confirmed exact-duration clip downloads and that one session
cannot search another session's index.

### Real neural integration results

The separate `.venv-ml` installed successfully and `pip check` reported no
broken requirements. Tested versions: WhisperX 3.8.6, Transformers 4.57.6,
sentence-transformers 5.7.0, PyTorch 2.8.0, pyannote.audio 4.0.7, NumPy 2.4.6.

| Check | Observed result |
|---|---|
| MiniLM real inference, including a long transcript | Normalized `(2, 384)` output; passed |
| LAION-CLAP real audio/text inference | Matching 512-dimensional vectors; passed |
| WhisperX `tiny.en` + forced alignment on CPU | 25 words from a 12-second speech excerpt; passed |
| Real upload through ASR, chunking, indexing, and retrieval | One searchable chunk, with UNKNOWN speaker and an explicit missing-token notice; passed |
| Entire demo indexed without either fallback | Text `(13, 384)`, audio `(13, 512)`; about 29.95 seconds |
| Neural interest-rate query | First result is the disagreement at 15.545 seconds |
| Neural renewable-energy query | First result includes battery storage at 75.434 seconds |
| Authenticated speaker diarization | Not run: token/model authorization required |

The live dashboard was also switched to the neural environment, re-indexed
through its UI with MiniLM + CLAP, and successfully played the top neural
result's exact 10.875-second clip.

These are integration checks, not accuracy benchmarks. In the real neural
run, the laughing/valuation query ranked a serious valuation discussion
first at the default weight. This is a genuine limitation of the model/demo,
not hidden or corrected by injecting transcript labels into audio vectors.
The offline lexical test's expected ranking is not evidence of CLAP's
laughter-recognition accuracy.

On this Windows machine, pyannote emits a TorchCodec/FFmpeg warning. The
WhisperX smoke test nevertheless completed because AudioSleuth supplies
decoded in-memory audio. Do not infer from that success that arbitrary
TorchCodec file decoding works; install an appropriate FFmpeg shared-library
runtime if using that separate path.

Gradio 5.x emits deprecation warnings about its upcoming 6.x APIs. These
warnings are not hidden and do not prevent the pinned-major application
from running.

## Session data, privacy, and deployment limits

- Each browser session has its own collection and index. Model objects may
  be reused, but transcripts and result matrices are not a global corpus.
- A bounded queue serializes ingestion/search callbacks; large model work
  does not spawn unlimited concurrent jobs.
- Generated source/snippet/manifest files live in uniquely named collection
  folders under the OS temporary directory.
- Successful replacement deletes only that session's previous collection.
  Session-state expiry is one hour. Gradio media caches have a one-hour
  age-based cleanup policy.
- Abrupt process termination can leave temporary files. These settings are
  not a legal retention guarantee, secure erasure mechanism, or substitute
  for operational storage policies.
- The default server is loopback-only and unauthenticated for local use.
  A hostile process or administrator on the same machine is outside the
  privacy boundary. Media URLs are not a substitute for authorization.
- Query/transcript HTML is escaped; tokens are read from the environment,
  and backend failure logs avoid printing token values.
- Hugging Face telemetry and pyannote metrics default to disabled in the
  neural inference paths, in addition to Gradio analytics being disabled.

Before hosting for real users, add authenticated per-user storage and media
authorization, TLS, quotas, a sandboxed media decoder, worker isolation,
job cancellation, robust persistence and deletion policies, observability,
dependency/security maintenance, and load testing. Evaluate WER, diarization
error rate, alignment error, retrieval nDCG/Recall@k, and acoustic-retrieval
accuracy on consented representative recordings. Do not deploy this local
demo publicly merely by changing the bind address.

## Regenerating the bundled demonstration

This is optional; the finished WAV, manifest, and index are already included.
On Windows with the David and Zira system voices:

```powershell
powershell.exe -NoProfile -File tools\synthesize_demo.ps1
.\.venv\Scripts\python.exe tools\build_demo.py
```

The script collects real TTS word-start events and assembles original speech
turns, silence, controlled gain, source hashes, and precomputed vectors.
Changing the script changes the source hash and requires rebuilding the index.
Intermediate turn files are written to `assets\_generated`; they are not
needed for runtime playback.

## Verified upstream API references

- [WhisperX usage and forced alignment](https://github.com/m-bain/whisperX#python-usage)
- [WhisperX pyannote wrapper](https://github.com/m-bain/whisperX/blob/main/whisperx/diarize.py)
- [Transformers CLAP API and LAION weights](https://huggingface.co/docs/transformers/v4.57.1/en/model_doc/clap)
- [Sentence-transformers](https://www.sbert.net/)
- [Gradio Audio](https://www.gradio.app/docs/gradio/audio)

Model weights and upstream libraries retain their own licenses and access
conditions. No third-party model weights or real podcast recordings are
redistributed in this project.
