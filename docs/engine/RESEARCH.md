# RESEARCH — spike findings

R&D to de-risk a symbolic-music composition & refinement engine. Each spike has an
explicit pass/fail bar. Updated as spikes complete.

| Spike | What it de-risks | Status | Verdict |
|---|---|---|---|
| A | Melody extraction quality (whistle/hum → notes) | ✅ harness done | **PASS (synthetic, provisional)** — recommend **pYIN**; needs real clips to confirm |
| B | Data model is lossless & editable | ✅ done | **PASS** (14/14) |
| C | Deterministic transforms are surgical/reversible/correct | ✅ done | **PASS** (22/22) |
| D | Constrained variation approach (Magenta vs LLM) | ✅ done | **GO with guardrails** (LLM + validate-repair) |
| E | Public-domain closest-match (stretch) | ✅ done | **PASS** (4/4) |

Environment: Python 3.11 (`uv`, `.venv`), TensorFlow-free engine; `basic-pitch` quarantined
in `.venv-bp`. See [`DECISIONS.md`](DECISIONS.md) §E for why.

---

## Spike B — data model round-trip → **PASS** (14/14)

Code: [`spikes/b_roundtrip.py`](spikes/b_roundtrip.py). Run: `uv run python spikes/b_roundtrip.py`.

Three deliberately-separated tests so MIDI's known lossiness can't masquerade as a schema bug:

1. **`model → JSON → model` (the contract):** byte-identical round-trip (sorted-key
   equality) on the **canonical form** (`exclude_none`), and the dump validates against
   `song.schema.json`. The rich layers are explicitly asserted present and preserved —
   `measured` (f0/cents/seconds), locks, provenance, roman-numeral harmony,
   constraints/motifs. A single-note edit propagates to export. **Lossless up to canonical
   normalization** — hand-authored JSON from the future TS frontend that emits explicit
   `null`s or redundant defaults (`tie: null`, `velocity: 80`) *normalizes* on first
   load (no information lost), rather than surviving byte-for-byte; that is correct, intended
   behavior.
2. **`model → MIDI → model` (musical layer only):** note count, pitches, and onset/duration
   ticks all preserved. Spelling, sections, roman numerals, measured, and locks are *not*
   in MIDI — expected and documented, which is exactly why JSON is canonical.
3. **`model → MusicXML → reload` (spelling):** B♭4 stays `B-4` and F♯4 stays `F#4` through
   MusicXML; the same pitches in MIDI are just `70`/`66` with no recoverable spelling.

**Risk retired:** the schema captures everything the product needs and survives the
export/import targets at the right fidelity per format.

---

## Spike C — deterministic transforms → **PASS** (22/22)

Code: [`spikes/c_transforms.py`](spikes/c_transforms.py), engine in
[`src/symbolic/transforms.py`](src/symbolic/transforms.py) + [`theory.py`](src/symbolic/theory.py).
Run: `uv run python spikes/c_transforms.py`. **This is the real test of whether the data
model supports refinement — and it does.**

- **Transpose (interval-based):** `A4` above C spells **F♯4**, `d5` above C spells **G♭4** —
  same pitch class (66), different spelling. Proves transpose is interval-correct, not
  semitone-guess. `P4` up then `−P4` down restores the symbolic signature exactly.
- **Locks honored:** a fully-locked note stays `C4` through transpose and keeps its
  `measured` data; moved notes have `measured` nulled (D3a) and a `transform:transpose`
  provenance tag, with **stable ids** and unchanged timing → diffs are surgical.
- **Mode swap (parallel ionian↔aeolian):** melody `E4`(deg 3) → `Eb4`, `D4`(deg 2)
  unchanged; chords `I → i` (pcs `[0,3,7]`) and `V → v` (pcs `[2,7,10]`); round-trip
  restores the signature. Invertibility is *tested*, not assumed.
- **Diatonic reharmonization:** produced `I iii I ii` on the demo. **Guarantee verified:**
  every strong-beat (slot-onset) melody note is a chord tone, every chord is diatonic, and
  the progression has real variety (not the trivial all-I). NCT treatment is defined
  (off-strong-beat notes may be passing/neighbor tones; a non-diatonic strong note resolves
  as an appoggiatura over tonic). Melody is untouched.
- Every transform output **re-validates against `song.schema.json`** — transforms keep the
  document legal.

**Risk retired:** the model supports surgical, reversible, theory-correct refinement.

---

## Spike A — melody extraction → **PASS (synthetic, provisional); recommend pYIN**

Code: [`spikes/a_extraction.py`](spikes/a_extraction.py), engine in
[`src/symbolic/extraction.py`](src/symbolic/extraction.py). Run: `uv run python spikes/a_extraction.py`.

Corpus: 3 melodies (scale, arpeggio, stepwise) × 2 timbres (pure-sine "whistle";
harmonic+vibrato+noise "voice") = 6 synthetic clips with exact ground truth. Scoring:
note-level F1 via `mir_eval`, onset tol 50 ms, pitch tol 50 cents, offsets ignored.

| tool | mean F1 | min F1 | s/clip | notes |
|---|---|---|---|---|
| **pYIN** (librosa) | **1.00** | 1.00 | **1.8** | pure-Python, no TF/torch; fastest; **recommended** |
| torchcrepe (full) | 1.00 | 1.00 | 4.2 | equally accurate; heavier (torch); good fallback for noisy real voice |
| basic-pitch | 0.61 | 0.08 | 1.7 | perfect on whistle, **collapses on voice timbre** |

All three recovered the identical key-invariant interval sequence `[2,2,1,2,2,2,1]` on the
clean scale.

**Failure mode (basic-pitch):** it is a *polyphonic, multi-instrument* note transcriber.
On a harmonically rich monophonic tone it detects overtones (2×/3×/4×) as extra notes,
tanking precision (F1 0.08–0.33 on "voice"). It also carries the TF cold-start and the
numpy-downgrade env tax (see §E2). Right tool for the wrong job here.

**Recommendation: pYIN** as the primary tracker for monophonic hum/whistle (perfect
accuracy on the corpus, ~2.4× faster than CREPE, no TF/torch), with **torchcrepe** kept as
a fallback — a learned model may prove more robust on breathy/drifting real voice. Drop
basic-pitch for monophonic input.

**Pass status — PROVISIONAL.** Synthetic is the easy case. The bar requires ≥2–3 **real**
hum/whistle clips; the harness scores any clip dropped in [`data/clips/`](data/clips) with a
paired `.gt.json` (see its README). Note the synthetic "voice" timbre already stresses the
trackers (it kills basic-pitch); pYIN/torchcrepe hold at 1.00, which is encouraging but not
a substitute for real input.

### First real clip — `whistle.m4a` (11 s, decoded via ffmpeg)
Run: `uv run python spikes/transcribe.py data/clips/whistle.m4a`. No ground truth yet, so
this is a qualitative read, but it overturned a synthetic-only assumption:
- **torchcrepe FAILED** — locked onto a ~58 Hz subharmonic (returned `A#1`, 4 notes). CREPE
  handled the *low* synthetic sine but a real *high* whistle (~B5–E6, 1–1.3 kHz) is out of
  its comfort zone. (It may still do well on hummed voice — in-distribution.)
- **pYIN with default `fmin=C2` had octave-error artifacts** — spurious `B2` at note
  boundaries (3 octaves below the fundamental).
- **Fix (now the default): register auto-detection.** A coarse pass finds the median voiced
  pitch and re-brackets `fmin`/`fmax` ±1 octave around it. With that, **pYIN returns a clean
  36-note melody** entirely in B5–E6 with only small, musical intervals (no octave jumps):
  `C6 B5 C6 B5 D#6 E6 C6 B5 C6 … D6 B5`. The result quantizes to a schema-valid `Song`
  (`spikes/_out/whistle_song.json`).
- basic-pitch: still octave-confused (±37–54 semitone jumps).

**Takeaway:** real data confirmed and *sharpened* the recommendation — **pYIN + auto-register**
is the tracker; torchcrepe is a poor fit for whistle specifically.

### Second real clip — `hum.m4a` (low voice, ~A2/110 Hz) + robustness hardening
A real hummed clip drove three concrete hardening fixes (all verified to leave the synthetic
corpus at **1.00**, i.e. no regression):

1. **torchcrepe recovers on hum** (it failed on whistle) — confirms CREPE is good on
   in-distribution voice, weak on whistle. pYIN stays the robust default across both.
2. **Vibrato over-segmentation (found, fixed).** A sustained vibrato note was being chopped
   into an alternating `G#2↔A2` semitone trill (~11×). Fix: **vibrato-robust hysteresis** in
   `frames_to_notes` — a note keeps its center pitch (running median of its frames) through
   wobble and only splits when the smoothed pitch moves ≥ 0.6 semitone for ≥ 4 consecutive
   frames; the note's pitch is the median of its frames, not a per-frame vote. Result on the
   hum: **27 → 11 notes**, trill gone.
3. **Edge-artifact trimming via the measured layer.** Extraction now records per-note
   `confidence` / `centsDev` / `f0Hz` (the two-layer fidelity, now populated from extraction).
   The breath/glottal `C2` boundary glitches measured **conf 0.067–0.075** vs **0.62–0.74**
   for real notes, so a `min_confidence=0.12` gate removes them cleanly: **11 → 7 notes**
   (`G#2 A#2 B2 A#2 G2 A#2 A2` — the actual phrase).

**Resolved — adaptive onset-aware segmentation.** The vibrato hysteresis alone merged
genuinely *repeated/rhythmic* notes (it smeared a contrabassoon trap riff into held notes).
Fix: split a held note at **strong re-attacks** — `librosa` onset detection gated to onsets
whose strength exceeds ~3× the active-frame median, and only when the current note has
already lasted ≥ `min_dur`. This is content-adaptive by construction: a rhythmic line
(trap bass, sharp attacks) splits correctly, while weak amplitude wobble inside a sustained
tone (vibrato hum, clean synth) stays *below* the strength gate and is merged. Verified
simultaneously: contrabassoon riff **24 → 34 notes** (rhythm restored), hum stays **7**
(vibrato still merged), synthetic stays **pYIN 1.00** (no regression).

**Still open:** a clip of a *known* melody (± a `.gt.json`) to get a *measured* real-clip F1
(both real clips so far are qualitative — validated by ear via `spikes/reconstruct.py`).

### Environment finding (also a result)
- `basic-pitch==0.4.0` cannot live in the engine env: it pulls **TensorFlow 2.15** and
  **downgrades numpy 2.4 → 1.26**, and its advertised `onnx` extra does not exist in 0.4.0.
  Quarantined in `.venv-bp`; needed `setuptools<81` there (0.4.2 `resampy` imports
  `pkg_resources`, removed in setuptools 81+). A concrete integration constraint to record.

---

## Spike D — constrained variation → **GO (with guardrails)**

Code: [`spikes/d_variation.py`](spikes/d_variation.py), engine in
[`src/symbolic/variation.py`](src/symbolic/variation.py). Run: `uv run python spikes/d_variation.py`.
Backend: local **Ollama** (no `CEREBRAS_API_KEY` was set), schema-enforced `format`. Task:
vary a 7-note C-major phrase; keep it in scale, in range, preserve the 3-note opening
motif, preserve total length. N=5 per model.

| metric (pre-repair unless noted) | qwen2.5:7b | llama3.1:8b |
|---|---|---|
| JSON parse_ok | 100% | 100% |
| **first-shot fully valid** | **0%** | **0%** |
| · in-key | 20% | 40% |
| · in-range | 100% | 100% |
| · motif preserved | 40% | 20% |
| · length preserved | 40% | 0% |
| post-repair valid | 100% | 100% |
| **usable AND distinct from input (post-repair)** | **100%** | **60%** |
| coherence: in-scale / smooth / novelty | 100 / 100 / 46% | 100 / 100 / 25% |
| mean latency (warm) | 2.5s | 2.1s |

Repaired variation → `Song` → **schema-valid: YES** (LLM output integrates with the model).

**The finding:** structured-JSON output is *reliable* (100% parse, range respected), but raw
LLM output is *not* reliable for hard musical constraints — both models routinely leave the
scale, drop the motif, and miscount total duration (first-shot fully-valid = 0%). A
**deterministic validate-and-repair loop** (snap-to-scale, clamp range, restore motif, fix
length) lifts this to 100% post-repair valid, in-key, schema-valid output.

**The honest yield is `usable AND distinct`** — post-repair valid *and* not a fallback copy
of the input (the repair can degenerate to identity if the model returns too few notes or an
unfixable length). On that stricter bar: **qwen2.5:7b = 100%** (every run a real variation),
**llama3.1:8b = 60%** (2 of 5 fell back to identity, tracking its 0% length / 20% motif
first-shot). So the GO rests on qwen's 100% usable-distinct yield, not the inflatable bare
100%. Latency ~2s warm locally; a hosted endpoint (Cerebras) would cut it further.

**Go/no-go: GO, with guardrails.** The viable architecture is *LLM-as-component inside a
symbolic validate-and-repair loop*, not LLM-as-oracle — the same thesis the whole project
rests on. Recommend **qwen2.5:7b** as the local default (best usable-distinct yield), keep a
hosted endpoint (Cerebras) as a latency/quality upgrade, and report `usable_distinct` as the
real quality KPI. Better prompting / constraint-aware decoding would raise first-shot rates
but is not required for usability.

**Magenta / MusicVAE:** not run live, per the agreed time-box. Classic MusicVAE needs TF1 /
Python 3.7–3.8 checkpoints; on this Py3.13/Win stack even the far newer basic-pitch (TF2)
needed a quarantined env + a `setuptools` pin (Spike A) — a TF1-era stack would cost
materially more to stand up for an approach the LLM path already covers. Recommend deferring
unless a specific need (e.g. smooth latent interpolation) arises; `note-seq` could be reused
without the legacy model.

## Vertical slice — end-to-end on real input (capabilities 1 + 2)

Code: [`spikes/vertical_slice.py`](spikes/vertical_slice.py), engine in
[`src/symbolic/analysis.py`](src/symbolic/analysis.py) + `io_render.render_arrangement`.
Run: `uv run python spikes/vertical_slice.py spikes/_out/whistle_song.json`.

The isolated spikes wired into one product loop on the **real clips**: extracted Song →
**detect key** → **melodic fingerprint** → **reharmonize in the detected key** → **render
melody+chords** → schema-valid Song out. This is the songwriting-assistant core (capability 1)
plus the measured fingerprint (capability 2, previously unbuilt), no UI.

- **Key detection** (music21 Krumhansl-Schmuckler) returns tonic+mode, a confidence
  (correlation), and ranked alternatives — honest about ambiguity. whistle → **A minor**
  (0.68; alts G major, C major); hum → **G minor** (0.357 — appropriately low for a 7-note
  phrase).
- **Fingerprint** (`analysis.py`) — every field a real computed feature: range, signed
  interval histogram, pitch-class histogram, Parsons contour, notes/sec density, mode,
  syncopation.
- **Reharmonization in the detected key** reported strong-beat chord-tone coverage **9/9**
  (whistle) and **4/6** (hum; the 2 misses are chromatic non-chord tones, expected on real
  input). Both arrangements are schema-valid and rendered to wav.

**Tempo/beat detection (done).** `extraction.estimate_tempo` seeds from `librosa.beat`,
then resolves the common tempo-octave (2x/half) ambiguity *by data* — it scores octave
variants + an inter-onset-interval estimate by how well the actual note onsets fit a
16th-note grid (preferring a musical 70-140 range). `transcribe.to_song` snaps onsets/
durations to that grid and writes the real `tempoMap`; raw timing stays in the measured
layer. Results: hum **103 BPM** (syncopation 0.86 → **0.43**), whistle **99 BPM** (correctly
halved from a 198.8 doubling; syncopation 0.97 → **0.71**, now a real measure of a fast loose
whistle). The arrangement now grooves to the detected tempo.

**Remaining timing caveats (future):** constant-tempo assumption (no rubato/tempo curve), and
the grid phase assumes the first note is on a downbeat. Both are fine for short phrases;
beat-time-anchored quantization (using `librosa` beat positions) would handle expressive
timing later.

---

## Spike E — public-domain matching (stretch) → **PASS** (4/4)

Code: [`spikes/e_matching.py`](spikes/e_matching.py), engine in
[`src/symbolic/similarity.py`](src/symbolic/similarity.py). Run: `uv run python spikes/e_matching.py`.

- Loaded **150 Essen Folksong tunes** straight from the `music21` core corpus — **public
  domain, no scraping, no copyrighted catalogs**. Legally clean by construction.
- **Self-retrieval:** a known tune ("Der Brautmoerder") returns *itself* as top-1 (n-gram
  similarity 1.0, DTW 0.0).
- **Key-invariance:** the same tune transposed up a fourth (+5 semitones) still retrieves
  itself — confirms the interval-based matching is transposition-invariant.
- **Novel query:** the demo phrase returns a ranked list of nearest folk tunes with scores.
- Similarity = interval **n-gram overlap** (Jaccard) + **DTW** on the interval contour.

**Risk retired:** capability 3 (closest public-domain match) is both technically and
legally clean. Scaling to the full Essen set (≈6 k tunes) and adding an inverted-index
pre-filter is straightforward future work.
