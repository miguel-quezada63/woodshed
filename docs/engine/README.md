# Symbolic-Music Composition & Refinement Engine — R&D

Turns hummed/whistled/played melodies into **editable musical structure**, then lets you
refine that music surgically — like editing a vector graphic, not re-prompting a black box.

**Core bet:** music lives as **symbolic data** (notes, intervals, chords, sections) as the
single source of truth. Audio is a disposable render off the bottom, never the artifact.
That is what makes coherent, diffable, reversible refinement possible.

> This repo is the **R&D / de-risking phase** — a validated data model plus working spikes.
> It is *not* the application. No UI, no audio generation, rendering is a stub.

## Status — all five spikes complete

| Spike | De-risks | Verdict |
|---|---|---|
| **A** | melody extraction (whistle/hum → notes) | PASS (synthetic, provisional) — **use pYIN**; add real clips to confirm |
| **B** | data model is lossless & editable | **PASS** 14/14 |
| **C** | deterministic transforms surgical/reversible/theory-correct | **PASS** 22/22 |
| **D** | constrained variation (LLM vs Magenta) | **GO** — LLM + validate-and-repair loop |
| **E** | public-domain closest-match (stretch) | **PASS** 4/4 |

See [`RESEARCH.md`](RESEARCH.md) for per-spike findings and [`DECISIONS.md`](DECISIONS.md)
for the data-model rationale.

## Layout

```
schema/song.schema.json     # the shared contract (language-agnostic, JSON Schema 2020-12)
src/symbolic/
  model.py                  # pydantic mirror of the schema (Python side of the contract)
  fixtures.py               # demo Song exercising every layer
  io_render.py              # MIDI (pretty_midi) + MusicXML (music21) import/export
  theory.py                 # music21 bridge: scales, diatonic triads, pitch conversion
  transforms.py             # transpose / mode-swap / diatonic reharmonization (pure fns)
  extraction.py             # pYIN / torchcrepe / basic-pitch -> notes -> intervals
  variation.py              # LLM constrained variation + validate-and-repair
  similarity.py             # Essen interval n-gram + DTW closest-match
spikes/                     # one runnable script per spike (a..e)
tools/bp_runner.py          # basic-pitch subprocess (runs in the quarantined .venv-bp)
data/clips/                 # drop real hum/whistle clips here (see its README)
```

## Setup & run

Requires [`uv`](https://docs.astral.sh/uv/) and Python 3.11 (the engine env is TF-free).

```bash
uv sync                          # create .venv and install deps
uv run python main.py            # fast path: spikes B + C (no network/ML)
uv run python main.py all        # everything (A needs ML deps, D needs Ollama/Cerebras)
uv run python spikes/c_transforms.py   # or run a single spike
```

**Real-clip pipeline (extraction → refine → render):**
```bash
uv run python spikes/transcribe.py     data/clips/whistle.m4a   # extract -> Song JSON
uv run python spikes/reconstruct.py    spikes/_out/whistle_song.json data/clips/whistle.m4a   # ear-check render
uv run python spikes/vertical_slice.py spikes/_out/whistle_song.json   # detect key + fingerprint + reharmonize + render
```

**Spike A (extraction)** also quarantines `basic-pitch` in a second env (it drags in
TensorFlow + a numpy downgrade — see DECISIONS §E2):
```bash
uv venv .venv-bp --python 3.11
uv pip install --python .venv-bp/Scripts/python.exe basic-pitch "setuptools<81"
```
Then drop 2–3 real monophonic clips in [`data/clips/`](data/clips) and re-run Spike A to
turn its provisional pass into a real one.

**Spike D (variation)** uses a local Ollama (`OLLAMA_HOST`) or Cerebras (`CEREBRAS_API_KEY`).
Keys/hosts come from env vars only.

## Constraints honored

No copyrighted melodies or recordings anywhere; the only corpus is the public-domain Essen
Folksong Collection bundled in `music21`. No audio-generation model. Symbolic correctness
over timbre.
