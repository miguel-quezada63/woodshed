"""Stem -> notes -> beat-grid-quantized Song.

Each stem is transcribed as a single line with the engine's pYIN extractor
(auto-register, vibrato-robust, onset-aware - see docs/engine/RESEARCH.md, Spike A).
That is the right tool for melodic stems (vocals / horn & flute leads in `other` /
bass). For chordal stems (piano, guitar) it only follows the top line.

Quantization runs against the DETECTED BEAT TIMES (not a constant tempo), so a
band that drifts still lands on the grid. Grids:
  16th    straight sixteenths
  8th     straight eighths (cleaner charts for busy lines)
  swing8  snaps to the swung off-beat (2/3 of the beat) and notates it as a plain
          eighth, the way jazz/electro-swing charts are written ("Swing 8ths")
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from symbolic import extraction
from symbolic.model import (Key, Measured, Note, Pitch, Provenance, Section, Song,
                            TempoEvent, TimeSignatureEvent, Track)

from . import fastpitch
from .analyze import Beats, KeyInfo

GRIDS = ("16th", "8th", "swing8")
PPQ = 960
# Per-stem pYIN settings. Default = engine auto-register (median +/- 1 octave).
# Stems re-articulate more than a hum, so re-attacks split notes at a lower onset
# ratio than the engine's 3.0 (tuned for vibrato hums). Bass gets a fixed range down
# to ~E1 and a longer frame so the low fundamentals are actually voiced.
PYIN_DEFAULT = {"onset_ratio": 2.0}
PYIN = {"bass": {"onset_ratio": 2.0, "fmin": 38.0, "fmax": 420.0, "frame_length": 4096}}


def transcribe_stem(wav: Path, stem: str, cache: Optional[Path] = None,
                    gate_db: float = 30.0,
                    on_progress: Optional[Callable[[float], None]] = None) -> List[Dict]:
    """-> [{midi, onset, offset, confidence, ...}] in seconds. Notes quieter than
    `gate_db` below the stem's loud passages are dropped (separation bleed).
    pYIN runs chunked across all CPU cores; `on_progress(0..1)` follows it."""
    if cache and cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    import librosa
    cfg = PYIN.get(stem, PYIN_DEFAULT)
    two_pass = "fmin" not in cfg               # auto-register = cheap coarse pass + real pass

    def chunk_done(pass_no, done, total):
        if on_progress:
            f = done / total
            on_progress((0.1 * f if pass_no == 1 else 0.1 + 0.9 * f) if two_pass else f)

    notes = extraction.extract_pyin(str(wav), pyin=fastpitch.ChunkedPyin(chunk_done), **cfg)

    y, sr = librosa.load(str(wav), sr=22050, mono=True)
    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    floor = float(np.percentile(rms, 95)) * 10 ** (-gate_db / 20)
    kept = []
    for n in notes:
        a, b = int(n["onset"] * sr / hop), max(int(n["offset"] * sr / hop), int(n["onset"] * sr / hop) + 1)
        if float(np.mean(rms[a:b])) >= floor:
            kept.append(n)
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(kept), encoding="utf-8")
    return kept


def _snap(x: float, grid: str) -> float:
    if grid == "swing8":
        b = math.floor(x)
        f = x - b
        c = min((0.0, 2 / 3, 1.0), key=lambda v: abs(f - v))
        return b + {0.0: 0.0, 2 / 3: 0.5, 1.0: 1.0}[c]
    step = 0.25 if grid == "16th" else 0.5
    return round(x / step) * step


def quantize(notes: List[Dict], beats: Beats, downbeat: int = 0, grid: str = "16th"):
    """-> list of (start_beat, dur_beat, note_dict), monophonic, start >= 0.
    `downbeat` = which detected beat (0-3) is beat 1 of a bar."""
    if not notes:
        return []
    step = 0.25 if grid == "16th" else 0.5
    on = beats.to_beats([n["onset"] for n in notes]) - downbeat
    off = beats.to_beats([n["offset"] for n in notes]) - downbeat
    if on.min() < 0:                                     # pickup: prepend whole bars
        shift = math.ceil(-on.min() / 4) * 4
        on, off = on + shift, off + shift

    events = []
    for n, a, b in zip(notes, on, off):
        s, e = _snap(float(a), grid), _snap(float(b), grid)
        events.append([s, max(e, s + step), n, float(b - a)])
    events.sort(key=lambda ev: (ev[0], -ev[3]))
    mono = []
    for ev in events:                                    # one note per grid slot (keep the longer)
        if mono and ev[0] == mono[-1][0]:
            continue
        mono.append(ev)
    for cur, nxt in zip(mono, mono[1:]):
        if cur[1] > nxt[0]:
            cur[1] = nxt[0]                              # no overlaps
        elif nxt[0] - cur[1] <= step:
            cur[1] = nxt[0]                              # close one-slot gaps (offset jitter)
    return [(s, e - s, n) for s, e, n, _ in mono if e - s > 0]


def build_song(title: str, key: KeyInfo, beats: Beats, parts: Dict[str, list],
               grid: str) -> Song:
    """parts = {stem: quantize(...) output} -> one Song, a melody track per stem."""
    prefer = "flat" if key.prefers_flats else "sharp"
    tracks, length = [], PPQ * 4
    for stem, events in parts.items():
        mnotes = []
        for i, (s, d, n) in enumerate(events):
            mnotes.append(Note(
                id=f"{stem}_{i}", startTick=int(round(s * PPQ)), durationTick=int(round(d * PPQ)),
                pitch=Pitch.from_midi(int(n["midi"]), prefer=prefer),
                provenance=Provenance(source="extracted", note=f"stem:{stem}"),
                measured=Measured(onsetSec=round(n["onset"], 4),
                                  durationSec=round(n["offset"] - n["onset"], 4),
                                  f0Hz=n.get("f0Hz"), centsDev=n.get("centsDev"),
                                  confidence=n.get("confidence"))))
        if mnotes:
            length = max(length, max(x.startTick + x.durationTick for x in mnotes))
        tracks.append(Track(id=f"trk_{stem}", kind="melody", voiceId=stem, role=stem,
                            instrumentHint=stem, notes=mnotes))
    bar = PPQ * 4
    length = int(math.ceil(length / bar) * bar)
    k = Key(tonic=key.tonic, mode=key.mode, confidence=key.confidence)
    return Song(
        id=f"song_{title[:40]}", title=title, key=k,
        tempoMap=[TempoEvent(atTick=0, bpm=beats.bpm)],
        timeSignatureMap=[TimeSignatureEvent(atTick=0, numerator=4, denominator=4)],
        sections=[Section(id="s0", type="other", label=f"grid:{grid}", startTick=0,
                          lengthTick=length, key=k, tracks=tracks)])
