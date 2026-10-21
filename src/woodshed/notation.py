"""Song -> readable sheet music.

music21 builds the score (measures, ties, rests, beaming, key/time signature,
tempo); Verovio engraves it to SVG for the browser. MusicXML is also written so
any part opens in MuseScore / Finale / Dorico for printing or editing.

Flute fit: each part is moved by whole octaves so its middle sits in the middle
of the staff, then any stragglers are folded into C4-G6 (playable AND readable -
fewer ledger lines than the full C4-C7 range). Flute is a C instrument, so no
transposition beyond octaves is ever needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import music21 as m21
import numpy as np

from symbolic.model import Song, Track
from symbolic.theory import pitch_to_m21

FLUTE_LOW, FLUTE_HIGH, FLUTE_CENTER = 60, 91, 74       # C4, G6, D5


def flute_shift(midis: List[int]) -> int:
    """Octave shift (semitones) that centers the line in flute range."""
    if not midis:
        return 0
    return int(round((FLUTE_CENTER - float(np.median(midis))) / 12)) * 12


def _fold(midi: int) -> int:
    while midi < FLUTE_LOW:
        midi += 12
    while midi > FLUTE_HIGH:
        midi -= 12
    return midi


def track_part(song: Song, track: Track, flute_fit: bool, swing: bool) -> Tuple[m21.stream.Part, str]:
    """-> (Part, display name)."""
    ppq = song.ppq
    notes = track.notes or []
    shift = flute_shift([n.pitch.midi for n in notes]) if flute_fit else 0
    stem = track.role or track.id
    if flute_fit:
        octs = shift // 12
        label = f"{stem.title()} -> Flute" + (f" ({'+' if octs > 0 else ''}{octs} oct)" if octs else "")
    else:
        label = stem.title()

    p = m21.stream.Part()
    p.id = stem
    p.partName = label
    inst = m21.instrument.Flute() if flute_fit else m21.instrument.Instrument()
    inst.partName = label
    p.insert(0, inst)
    sec = song.sections[0]
    key = sec.key or song.key
    p.insert(0, m21.key.Key(key.tonic, "minor" if key.mode in ("minor", "aeolian") else "major"))
    p.insert(0, m21.meter.TimeSignature("4/4"))
    p.insert(0, m21.tempo.MetronomeMark(number=round(song.tempo_bpm())))
    if swing:
        p.insert(0, m21.expressions.TextExpression("Swing 8ths"))

    for n in notes:
        pitch = pitch_to_m21(n.pitch)
        if flute_fit:              # move the octave number itself: spelling never changes
            pitch.octave += (_fold(n.pitch.midi + shift) - n.pitch.midi) // 12
        m21n = m21.note.Note(pitch, quarterLength=n.durationTick / ppq)
        p.insert((sec.startTick + n.startTick) / ppq, m21n)

    if not flute_fit:
        midis = [n.pitch.midi for n in notes]
        if midis and float(np.median(midis)) < 55:
            p.insert(0, m21.clef.BassClef())
    end_q = (sec.startTick + sec.lengthTick) / ppq
    p.makeRests(fillGaps=True, inPlace=True, timeRangeFromBarDuration=False)
    if p.highestTime < end_q:                          # pad the last bar
        p.insert(p.highestTime, m21.note.Rest(quarterLength=end_q - p.highestTime))
    return p, label


def build_score(song: Song, stems: Optional[List[str]] = None, flute_fit: bool = True,
                swing: bool = False) -> m21.stream.Score:
    score = m21.stream.Score()
    md = m21.metadata.Metadata()
    md.title = song.title or "Transcription"
    md.composer = "transcribed with woodshed"
    score.insert(0, md)
    for trk in song.sections[0].tracks:
        if stems and (trk.role or trk.id) not in stems:
            continue
        part, _ = track_part(song, trk, flute_fit, swing)
        score.insert(0, part)
    return score


def write_musicxml(score: m21.stream.Score, path: Path) -> Path:
    score.write("musicxml", fp=str(path))
    return path


def write_midi(score: m21.stream.Score, path: Path) -> Path:
    score.write("midi", fp=str(path))
    return path


def render_svg(musicxml_path: Path, width: int = 1900, scale: int = 50) -> List[str]:
    """MusicXML -> list of SVG pages (Verovio)."""
    import verovio
    tk = verovio.toolkit()
    tk.setOptions({"pageWidth": width, "pageHeight": 2970, "scale": scale,
                   "adjustPageHeight": True, "breaks": "auto", "footer": "none",
                   "pageMarginLeft": 60, "pageMarginRight": 60})
    if not tk.loadFile(str(musicxml_path)):
        raise RuntimeError(f"Verovio could not load {musicxml_path}")
    return [tk.renderToSVG(i) for i in range(1, tk.getPageCount() + 1)]
