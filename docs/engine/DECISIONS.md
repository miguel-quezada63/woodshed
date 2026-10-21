# DECISIONS — symbolic data model & architecture

Status: R&D phase. The data model is implemented and validated by Spikes B and C.
Schema of record: [`schema/song.schema.json`](schema/song.schema.json). Python mirror:
[`src/symbolic/model.py`](src/symbolic/model.py).

---

## Thesis being protected

Music is symbolic data (notes, intervals, chords, sections) as the **single source of
truth**. Audio is a disposable render off the bottom, never the artifact. Everything in
this model exists to make refinement **coherent, diffable, reversible, and surgical** —
the way you edit a vector graphic, not re-prompt a black box.

Two properties carry that thesis, and every decision below serves them:

1. **Stable IDs on every element** → diffs match *semantically* (by id), not positionally.
2. **Value semantics** (the document is a pure snapshot, no embedded behavior/history) →
   undo = keep the previous snapshot; transforms are pure `Song → Song`.

---

## Decisions

### D1 — Canonical timing is **integer ticks** at `meta.ppq` (default 960)
- `quarterLength` is *derived* (`tick / ppq`); `seconds` live only in the measured layer.
- **Why 960:** common tuplets divide evenly (triplet 320, 16th 240, quintuplet 192), so
  there is no fractional drift. This makes "unchanged element ⇔ byte-identical
  serialization" *exact*, which is what makes the diffable/reversible thesis real.
- **Rejected — float `quarterLength` as canonical:** a triplet eighth = 0.333… is not
  representable; error accumulates and untouched fields re-serialize differently, forcing
  an equality epsilon that quietly weakens diff/undo. Also TS has no `Fraction`, so floats
  hurt the shared contract too.
- **Rejected — seconds as canonical:** theory transforms and quantization need musical
  time; performed seconds are kept in `measured` instead.
- Verified by Spike B (byte-identical round-trip) and Spike C (surgical diffs).

### D2 — Pitch source of truth is **spelling** (`step` + `alter` + `octave`); `midi` is derived
- A `model_validator` enforces `midi == f(step, alter, octave)` and a consistent `name`, so
  the redundant convenience fields can never silently drift.
- **Why:** G♯ ≠ A♭ for key, mode, and reharmonization. MIDI numbers cannot spell.
- **Rejected — MIDI-only:** loses enharmonic correctness; Spike C's A4-vs-d5 test
  (F♯ vs G♭, same pitch class) would be impossible. MIDI kept only as a derived field +
  render/export target.
- Verified by Spike B (MusicXML keeps B♭, MIDI cannot) and Spike C (interval spelling).

### D3 — **Two-layer fidelity**: clean quantized notes are truth; raw performance in `measured`
- Each note may carry `measured { onsetSec, durationSec, f0Hz, centsDev, confidence }`.
- **Why:** the product's "measured fingerprint" pillar needs real numbers traceable to the
  performance, while transforms need a clean 12-TET grid to operate on.
- **Consequence (D3a):** any transform that changes a note's pitch/timing **nulls
  `measured`** (the performance no longer corresponds). Therefore **reversibility is scoped
  to symbolic fields only** — `transpose(+P4)` then `transpose(−P4)` restores pitch/timing/
  ids/structure, not the discarded raw audio numbers. This is stated, not hidden.
- **Rejected — clean-only:** fingerprint could only describe the cleaned-up version.
- **Rejected — raw-primary:** transforms and MIDI/DAW export get materially messier.

### D4 — **Symbols-primary harmony**: abstract chords are truth, voicing is derived/optional
- A `Chord` is `romanNumeral` + `chordSymbol` + `root` + `quality` + `chordPitchClasses` +
  `function`; `voicing` (explicit MIDI pitches) is `null` until a revoice op realizes it.
- **Why:** reharmonization operates on symbols/function; `chordPitchClasses` lets us test
  melody-note membership (the reharmonization correctness guarantee). Voicing is a later,
  separable concern.
- **Rejected — fully-voiced-as-truth:** heavier schema, more to get right this phase.
- **Rejected — chord-labels-only:** too thin for reharmonization/revoice to operate on.

### D5 — **Sections own tracks**; note/chord positions are **section-relative**
- `Section.startTick` is absolute song time; a note's `startTick` is an offset within its
  section. Cross-section voice identity is preserved via `Track.voiceId`.
- **Why:** section-level regeneration and locking are core ops; a self-contained,
  relocatable section is the cleanest unit to freeze, swap, or regenerate. Matches the
  stated `song → sections → tracks → notes/chords` hierarchy.
- **Rejected — song-owns-tracks, sections as time-markers:** simpler addressing but makes
  "regenerate this block while honoring neighbors" awkward; loses the natural lock unit.

### D6 — **Locks** = freeze flags, hierarchical, with optional partial dimensions
- `Lock { locked, dimensions? }` on section/track/note/chord. `locked=true` ⇒ hard
  constraint: transforms/generators must not alter it; section-regen honors it as context.
  `dimensions` (e.g. `["pitch"]`) allows partial freezes (freeze pitch, allow rhythm).
- **Why:** the vector-graphic "lock a layer" gesture, plus finer control for refinement.
- Verified by Spike C (a fully-locked note survives transpose with its `measured` intact).

### D7 — **Constraints** are distinct from locks
- `Constraints { scalePitchClasses, pitchRange, harmonicRhythmTicks, motifs[],
  referenceSections[], styleHints }` describe the *allowed space*; locks freeze *concrete
  material*. Section-regeneration and constrained variation (Spike D) consume constraints;
  motifs reference notes by id so "honor the motif" is explicit.

### D8 — Diff/undo via **immutable snapshots + ids + lightweight `provenance`**, not embedded history
- `Provenance { source: extracted|user|transform:<n>|generated:<model>, parentId?, note? }`.
- **Why:** keeps the shared contract a clean *value*; history is the app's job (a sequence
  of snapshots). Embedding an undo stack would bloat the contract with app state.

### D9 — **JSON (+ JSON Schema) is canonical**; MIDI/MusicXML are import/render targets only
- **Why:** MIDI/MusicXML cannot losslessly carry locks, constraints, provenance, the
  measured layer, or roman numerals. JSON is human-diffable and frontend-native.
- `additionalProperties: false` throughout — the contract is strict so typos/drift are
  caught; forward-compat is handled by `schemaVersion` (semver; frontend/engine gate on
  major). Verified: the Python model and the JSON Schema are cross-checked every spike (a
  model dump must validate against the schema), so the two contracts cannot drift apart.

---

## Environment decisions (Windows / Python 3.13 reality)

### E1 — Engine env is **Python 3.11** via `uv` (`.venv`), kept TensorFlow-free
- `music21`, `pretty_midi`, `librosa` (incl. `librosa.pyin`), `pydantic`, `jsonschema`,
  `torch`/`torchcrepe`, `onnxruntime` all install cleanly on numpy 2.x.

### E2 — `basic-pitch` is **quarantined** in a separate venv (`.venv-bp`)
- `basic-pitch==0.4.0` hard-depends on **TensorFlow 2.15** and would **downgrade numpy
  2.4 → 1.26**, polluting the engine env (its advertised onnx extra does not exist in
  0.4.0). It is therefore installed in an isolated venv and invoked as a subprocess that
  emits note JSON, so it gets a fair evaluation without contaminating the clean engine.
- See Spike A in [`RESEARCH.md`](RESEARCH.md) for the head-to-head result.

### E3 — Spike D leans **LLM-as-symbolic-generator**; Magenta is time-boxed
- Classic Magenta/MusicVAE is unmaintained on Python 3.13 (TF1 / Py3.7–3.8). Per the
  agreed plan, the LLM path (Cerebras / local Ollama, structured JSON in/out with a
  validate-and-repair loop) is the primary track; Magenta gets a capped attempt only.
