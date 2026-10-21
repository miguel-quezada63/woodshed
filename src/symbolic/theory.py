"""Thin music21 bridge: scales, diatonic triads, and pitch conversion.
Kept separate so transforms.py reads as music operations, not m21 plumbing.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import music21 as m21

from .model import Pitch, ChordRoot

_SCALE_CLASS = {
    "major": m21.scale.MajorScale, "ionian": m21.scale.MajorScale,
    "minor": m21.scale.MinorScale, "aeolian": m21.scale.MinorScale,
    "dorian": m21.scale.DorianScale, "phrygian": m21.scale.PhrygianScale,
    "lydian": m21.scale.LydianScale, "mixolydian": m21.scale.MixolydianScale,
    "locrian": m21.scale.LocrianScale,
    "harmonicMinor": m21.scale.HarmonicMinorScale,
    "melodicMinor": m21.scale.MelodicMinorScale,
}

# roman numerals by scale degree for the two qualities we reharmonize over
_ROMAN_MAJOR = {1: "I", 2: "ii", 3: "iii", 4: "IV", 5: "V", 6: "vi", 7: "vii°"}
_ROMAN_MINOR = {1: "i", 2: "ii°", 3: "III", 4: "iv", 5: "v", 6: "VI", 7: "VII"}
_FUNCTION = {1: "tonic", 2: "predominant", 3: "tonic", 4: "subdominant",
             5: "dominant", 6: "tonic", 7: "dominant"}


def ours_name_to_m21(name: str) -> str:
    """'Bb' -> 'B-', 'F#' -> 'F#' (music21 uses '-' for flats)."""
    return name.replace("b", "-")


def m21_to_pitch(p: m21.pitch.Pitch) -> Pitch:
    return Pitch.from_spelling(p.step, int(p.alter), int(p.octave))


def pitch_to_m21(p: Pitch) -> m21.pitch.Pitch:
    mp = m21.pitch.Pitch(step=p.step, octave=p.octave)
    if p.alter:
        mp.accidental = m21.pitch.Accidental(p.alter)
    return mp


def make_scale(tonic: str, mode: str) -> m21.scale.ConcreteScale:
    cls = _SCALE_CLASS.get(mode, m21.scale.MajorScale)
    return cls(ours_name_to_m21(tonic))


@dataclass
class Triad:
    degree: int
    roman: str
    quality: str          # major | minor | diminished | augmented
    function: str
    root: ChordRoot
    pitch_classes: List[int]


def _quality(root_pc: int, third_pc: int, fifth_pc: int) -> str:
    t = (third_pc - root_pc) % 12
    f = (fifth_pc - root_pc) % 12
    return {(4, 7): "major", (3, 7): "minor", (3, 6): "diminished",
            (4, 8): "augmented"}.get((t, f), "other")


def diatonic_triads(tonic: str, mode: str) -> Dict[int, Triad]:
    """Triad built on each scale degree (1..7) of the given key."""
    sc = make_scale(tonic, mode)
    is_minorish = mode in ("minor", "aeolian", "harmonicMinor", "melodicMinor",
                           "dorian", "phrygian", "locrian")
    romans = _ROMAN_MINOR if is_minorish else _ROMAN_MAJOR
    out: Dict[int, Triad] = {}
    for deg in range(1, 8):
        r = sc.pitchFromDegree(deg)
        third = sc.pitchFromDegree(deg + 2)
        fifth = sc.pitchFromDegree(deg + 4)
        rpc, tpc, fpc = r.pitchClass, third.pitchClass, fifth.pitchClass
        out[deg] = Triad(
            degree=deg, roman=romans[deg], quality=_quality(rpc, tpc, fpc),
            function=_FUNCTION[deg],
            root=ChordRoot(pitchClass=rpc, name=r.name.replace("-", "b")),
            pitch_classes=sorted({rpc, tpc, fpc}),
        )
    return out


def scale_pitch_classes(tonic: str, mode: str) -> List[int]:
    sc = make_scale(tonic, mode)
    pcs = {p.pitchClass for p in sc.getPitches(f"{ours_name_to_m21(tonic)}2",
                                               f"{ours_name_to_m21(tonic)}3")}
    return sorted(pcs)


def chord_symbol(root_name: str, quality: str) -> str:
    suffix = {"major": "", "minor": "m", "diminished": "dim", "augmented": "aug"}.get(quality, "")
    return f"{root_name}{suffix}"
