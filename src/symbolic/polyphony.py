"""Polyphonic path: polyphonic transcription (basic-pitch) -> phantom-harmonic
filter -> voice separation -> multi-track Song. No source separation needed to
start; the symbolic post-processing is where the real work (and our advantage) is.

The phantom filter exploits the probe finding (spikes/poly_probe.py): basic-pitch
OVER-detects, reporting real notes plus their strong harmonics. A note that sits a
harmonic interval above a LOUDER, CO-ONSET note (no independent attack) is almost
certainly that note's overtone, not a real voice. A genuine octave double, by
contrast, tends to have an independent onset or comparable strength - so we keep
those.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from .model import (Song, Section, Track, Note, Pitch, Key, Provenance, Measured,
                    TempoEvent)
from . import extraction

# semitones of harmonic k above the fundamental: round(12*log2(k)) for k=2..8
HARMONIC_SEMITONES = {12, 19, 24, 28, 31, 34, 36}


def transcribe_poly(wav_path: str) -> List[Dict]:
    """All simultaneous notes via basic-pitch: [{midi,onset,offset,amp}, ...]."""
    return extraction.extract_basic_pitch(wav_path)


def filter_harmonics(notes: List[Dict], onset_tol: float = 0.06,
                     amp_ratio: float = 0.9) -> Tuple[List[Dict], int]:
    """Drop notes that are a harmonic interval above a louder, co-onset note."""
    kept, removed = [], 0
    for n in notes:
        phantom = False
        for low in notes:
            if low is n or low["midi"] >= n["midi"]:
                continue
            if (n["midi"] - low["midi"]) in HARMONIC_SEMITONES \
                    and abs(n["onset"] - low["onset"]) <= onset_tol \
                    and n.get("amp", 1.0) < low.get("amp", 1.0) * amp_ratio:
                phantom = True
                break
        if phantom:
            removed += 1
        else:
            kept.append(n)
    return kept, removed


def separate_voices(notes: List[Dict]) -> Dict[str, List[Dict]]:
    """Skyline melody (top), bass (bottom), inner (the rest), by who is highest/
    lowest among the notes overlapping each note in time."""
    out = {"lead": [], "bass": [], "inner": []}
    for n in notes:
        overlap = [m["midi"] for m in notes
                   if m["onset"] < n["offset"] and m["offset"] > n["onset"]]
        if not overlap:
            out["lead"].append(n)
        elif n["midi"] == max(overlap):
            out["lead"].append(n)
        elif n["midi"] == min(overlap):
            out["bass"].append(n)
        else:
            out["inner"].append(n)
    return out


def _quantize_track(track_notes, t0, bpm, ppq, grid, voice):
    beat = 60.0 / bpm
    snap = max(1, ppq // grid)
    out = []
    for i, n in enumerate(sorted(track_notes, key=lambda x: x["onset"])):
        start = int(round((n["onset"] - t0) / beat * ppq / snap) * snap)
        dur = int(max(snap, round((n["offset"] - n["onset"]) / beat * ppq / snap) * snap))
        out.append(Note(
            id=f"{voice}_{i}", startTick=start, durationTick=dur,
            pitch=Pitch.from_midi(n["midi"], prefer="sharp"),
            velocity=int(max(1, min(127, round(n.get("amp", 0.6) * 127)))),
            provenance=Provenance(source="generated:poly-transcription"),
            measured=Measured(onsetSec=round(n["onset"], 4),
                              durationSec=round(n["offset"] - n["onset"], 4))))
    return out


def to_multitrack_song(voices: Dict[str, List[Dict]], wav_path: str, stem: str = "poly",
                       ppq: int = 960, grid: int = 4) -> Song:
    """Build a multi-track Song (one melody-kind track per voice). Uses the existing
    tracks[] schema - no changes needed."""
    all_notes = [n for v in voices.values() for n in v]
    bpm, _ = extraction.estimate_tempo(wav_path, notes=all_notes)
    t0 = min((n["onset"] for n in all_notes), default=0.0)

    tracks = []
    for voice in ("lead", "inner", "bass"):
        vn = voices.get(voice) or []
        if not vn:
            continue
        notes = _quantize_track(vn, t0, bpm, ppq, grid, voice)
        tracks.append(Track(id=f"trk_{voice}", kind="melody", voiceId=voice,
                            role=voice, instrumentHint=voice, notes=notes))
    length = max((n.startTick + n.durationTick for t in tracks for n in t.notes), default=ppq)
    return Song(
        id=f"song_{stem}", title=f"{stem} (polyphonic)", key=Key(tonic="C", mode="ionian"),
        tempoMap=[TempoEvent(atTick=0, bpm=bpm)],
        sections=[Section(id="s", type="phrase", startTick=0, lengthTick=length, tracks=tracks)])
