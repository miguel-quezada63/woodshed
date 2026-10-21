"""Key detection + the measured melodic fingerprint (product capability 2).

Every field is a real computed feature, not a vibe. Key detection uses music21's
Krumhansl-Schmuckler key-finding; the fingerprint is computed directly from the
symbolic notes (+ measured timing when present).
"""
from __future__ import annotations

from collections import Counter
from typing import List, Tuple

import music21 as m21

from .model import Song, Analysis, AnalysisRange, Key, Note
from .theory import pitch_to_m21


def _abs_melody(song: Song) -> List[Tuple[int, Note]]:
    items = [(sec.startTick + n.startTick, n) for sec, _, n in song.iter_melody_notes()]
    items.sort(key=lambda x: x[0])
    return items


def detect_key(song: Song) -> Key:
    """Krumhansl-Schmuckler key estimate from the melody. Returns a Key with the
    correlation coefficient as confidence."""
    s = m21.stream.Stream()
    for _, n in _abs_melody(song):
        s.append(m21.note.Note(pitch_to_m21(n.pitch),
                               quarterLength=max(0.25, n.durationTick / song.ppq)))
    if not s.notes:
        return song.key
    k = s.analyze("key")
    mode = k.mode if k.mode in ("major", "minor") else "major"
    conf = float(getattr(k, "correlationCoefficient", 0.0) or 0.0)
    return Key(tonic=k.tonic.name.replace("-", "b"), mode=mode, confidence=round(conf, 3))


def key_alternatives(song: Song, top: int = 3) -> List[Tuple[str, float]]:
    """Top alternative key interpretations (name, correlation) - useful because a
    short hummed melody is often ambiguous."""
    s = m21.stream.Stream()
    for _, n in _abs_melody(song):
        s.append(m21.note.Note(pitch_to_m21(n.pitch),
                               quarterLength=max(0.25, n.durationTick / song.ppq)))
    if not s.notes:
        return []
    k = s.analyze("key")
    out = [(f"{k.tonic.name} {k.mode}", round(float(k.correlationCoefficient), 3))]
    for alt in (k.alternateInterpretations or [])[: top - 1]:
        out.append((f"{alt.tonic.name} {alt.mode}", round(float(alt.correlationCoefficient), 3)))
    return out


def compute_fingerprint(song: Song, bpm: float = None) -> Analysis:
    items = _abs_melody(song)
    if not items:
        return Analysis(analysisVersion="0.1.0")
    starts = [t for t, _ in items]
    notes = [n for _, n in items]
    midis = [n.pitch.midi for n in notes]
    intervals = [midis[i + 1] - midis[i] for i in range(len(midis) - 1)]

    # rhythmic density: prefer real measured timing, else musical time
    bpm = bpm or song.tempo_bpm()
    if all(n.measured and n.measured.onsetSec is not None for n in notes):
        span = (notes[-1].measured.onsetSec + (notes[-1].measured.durationSec or 0)) \
            - notes[0].measured.onsetSec
    else:
        last_end = max(s + n.durationTick for s, n in items) - starts[0]
        span = (last_end / song.ppq) * (60.0 / bpm)
    nps = round(len(notes) / span, 3) if span > 0 else 0.0

    # syncopation: fraction of onsets that don't land on a quarter-note beat
    off_beat = sum(1 for s in starts if s % song.ppq != 0)
    syncopation = round(off_beat / len(starts), 3)

    return Analysis(
        analysisVersion="0.1.0",
        range=AnalysisRange(minMidi=min(midis), maxMidi=max(midis),
                            semitones=max(midis) - min(midis)),
        intervalHistogram=dict(Counter(str(d) for d in intervals)),
        pitchClassHistogram=dict(Counter(str(m % 12) for m in midis)),
        contour=["U" if d > 0 else "D" if d < 0 else "R" for d in intervals],
        rhythmicDensityNps=nps,
        detectedMode=song.key.mode,
        syncopation=syncopation,
    )


def annotate(song: Song) -> Song:
    """Pure: return a copy with detected key applied (song + sections) and the
    melodic fingerprint computed into song.analysis."""
    out = song.model_copy(deep=True)
    k = detect_key(out)
    out.key = k
    for sec in out.sections:
        if sec.key is None or True:   # adopt detected key as the working tonal center
            sec.key = k.model_copy()
    out.analysis = compute_fingerprint(out)
    return out
