"""Pydantic models mirroring schema/song.schema.json (the shared contract).

Timing is canonical in INTEGER TICKS at meta.ppq. quarterLength is derived
(tick / ppq). Pitch spelling (step/alter/octave) is the source of truth for
theory; midi/name are derived and kept in sync (enforced by a validator).

These models are the Python side of the contract; the JSON Schema is the
language-agnostic side. spike B validates that a model dump conforms to the
schema, so the two cannot silently drift.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator

STEP_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# default spelling tables (used only by from_midi convenience helpers)
_SHARP = {0: ("C", 0), 1: ("C", 1), 2: ("D", 0), 3: ("D", 1), 4: ("E", 0),
          5: ("F", 0), 6: ("F", 1), 7: ("G", 0), 8: ("G", 1), 9: ("A", 0),
          10: ("A", 1), 11: ("B", 0)}
_FLAT = {0: ("C", 0), 1: ("D", -1), 2: ("D", 0), 3: ("E", -1), 4: ("E", 0),
         5: ("F", 0), 6: ("G", -1), 7: ("G", 0), 8: ("A", -1), 9: ("A", 0),
         10: ("B", -1), 11: ("B", 0)}

Mode = Literal["major", "minor", "ionian", "dorian", "phrygian", "lydian",
               "mixolydian", "aeolian", "locrian", "harmonicMinor", "melodicMinor"]


def accidental_str(alter: int) -> str:
    if alter > 0:
        return "#" * alter
    if alter < 0:
        return "b" * (-alter)
    return ""


def spell_name(step: str, alter: int, octave: int) -> str:
    return f"{step}{accidental_str(alter)}{octave}"


def midi_from_spelling(step: str, alter: int, octave: int) -> int:
    return (octave + 1) * 12 + STEP_PC[step] + alter


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Meta(_Base):
    ppq: int = 960
    timeUnit: Literal["tick"] = "tick"
    pitchRef: Optional[dict] = None
    createdBy: Optional[str] = None


class Key(_Base):
    tonic: str
    mode: Mode
    confidence: Optional[float] = None


class TempoEvent(_Base):
    atTick: int
    bpm: float


class TimeSignatureEvent(_Base):
    atTick: int
    numerator: int
    denominator: int


class Lock(_Base):
    locked: bool = False
    dimensions: Optional[List[Literal["pitch", "rhythm", "harmony", "contour", "voicing"]]] = None

    def freezes(self, dimension: str) -> bool:
        """True if this lock forbids changing `dimension`."""
        if not self.locked:
            return False
        if not self.dimensions:        # fully frozen
            return True
        return dimension in self.dimensions


class Provenance(_Base):
    source: str
    parentId: Optional[str] = None
    note: Optional[str] = None


class Pitch(_Base):
    midi: int = Field(ge=0, le=127)
    step: Literal["A", "B", "C", "D", "E", "F", "G"]
    alter: int = Field(ge=-2, le=2)
    octave: int = Field(ge=-1, le=9)
    name: str

    @model_validator(mode="after")
    def _consistent(self) -> "Pitch":
        want_midi = midi_from_spelling(self.step, self.alter, self.octave)
        if want_midi != self.midi:
            raise ValueError(
                f"pitch midi {self.midi} != spelling {self.name} "
                f"(={want_midi}); spelling is source of truth")
        want_name = spell_name(self.step, self.alter, self.octave)
        if want_name != self.name:
            raise ValueError(f"pitch name {self.name!r} != spelled {want_name!r}")
        return self

    @classmethod
    def from_spelling(cls, step: str, alter: int, octave: int) -> "Pitch":
        return cls(midi=midi_from_spelling(step, alter, octave), step=step,
                   alter=alter, octave=octave, name=spell_name(step, alter, octave))

    @classmethod
    def from_midi(cls, midi: int, prefer: str = "sharp") -> "Pitch":
        table = _SHARP if prefer == "sharp" else _FLAT
        step, alter = table[midi % 12]
        octave = midi // 12 - 1
        return cls.from_spelling(step, alter, octave)

    @property
    def pitch_class(self) -> int:
        return self.midi % 12


class Measured(_Base):
    onsetSec: Optional[float] = None
    durationSec: Optional[float] = None
    f0Hz: Optional[float] = None
    centsDev: Optional[float] = None
    confidence: Optional[float] = None


class Note(_Base):
    id: str
    startTick: int
    durationTick: int
    pitch: Pitch
    velocity: int = 80
    tie: Optional[Literal["start", "stop", "continue"]] = None
    lock: Lock = Field(default_factory=Lock)
    provenance: Optional[Provenance] = None
    measured: Optional[Measured] = None


class ChordRoot(_Base):
    pitchClass: int = Field(ge=0, le=11)
    name: str


class Chord(_Base):
    id: str
    startTick: int
    durationTick: int
    romanNumeral: Optional[str] = None
    chordSymbol: Optional[str] = None
    root: Optional[ChordRoot] = None
    quality: Optional[str] = None
    inversion: Optional[int] = None
    chordPitchClasses: Optional[List[int]] = None
    function: Optional[Literal["tonic", "subdominant", "dominant", "predominant", "other"]] = None
    voicing: Optional[List[int]] = None
    lock: Lock = Field(default_factory=Lock)
    provenance: Optional[Provenance] = None


class Motif(_Base):
    id: str
    noteIds: List[str]
    label: Optional[str] = None


class PitchRange(_Base):
    minMidi: int
    maxMidi: int


class Constraints(_Base):
    scalePitchClasses: Optional[List[int]] = None
    pitchRange: Optional[PitchRange] = None
    harmonicRhythmTicks: Optional[int] = None
    motifs: Optional[List[Motif]] = None
    referenceSections: Optional[List[str]] = None
    styleHints: Optional[str] = None


class Track(_Base):
    id: str
    kind: Literal["melody", "harmony"]
    voiceId: Optional[str] = None
    role: Optional[str] = None
    instrumentHint: Optional[str] = None
    lock: Lock = Field(default_factory=Lock)
    notes: Optional[List[Note]] = None
    chords: Optional[List[Chord]] = None

    @model_validator(mode="after")
    def _kind_payload(self) -> "Track":
        if self.kind == "melody" and self.notes is None:
            raise ValueError("melody track requires notes[]")
        if self.kind == "harmony" and self.chords is None:
            raise ValueError("harmony track requires chords[]")
        return self


class AnalysisRange(_Base):
    minMidi: Optional[int] = None
    maxMidi: Optional[int] = None
    semitones: Optional[int] = None


class Analysis(_Base):
    analysisVersion: Optional[str] = None
    range: Optional[AnalysisRange] = None
    intervalHistogram: Optional[dict] = None
    pitchClassHistogram: Optional[dict] = None
    contour: Optional[List[Literal["U", "D", "R"]]] = None
    rhythmicDensityNps: Optional[float] = None
    detectedMode: Optional[str] = None
    syncopation: Optional[float] = None


class Section(_Base):
    id: str
    type: Literal["intro", "verse", "prechorus", "chorus", "bridge", "outro",
                  "instrumental", "phrase", "other"]
    label: Optional[str] = None
    startTick: int
    lengthTick: int
    key: Optional[Key] = None
    lock: Lock = Field(default_factory=Lock)
    constraints: Optional[Constraints] = None
    tracks: List[Track]
    analysis: Optional[Analysis] = None


class Song(_Base):
    schemaVersion: str = "0.1.0"
    id: str
    title: Optional[str] = None
    meta: Meta = Field(default_factory=Meta)
    key: Key
    tempoMap: Optional[List[TempoEvent]] = None
    timeSignatureMap: Optional[List[TimeSignatureEvent]] = None
    sections: List[Section]
    analysis: Optional[Analysis] = None

    # ---- convenience ----
    @property
    def ppq(self) -> int:
        return self.meta.ppq

    def tempo_bpm(self) -> float:
        if self.tempoMap:
            return self.tempoMap[0].bpm
        return 120.0

    def iter_melody_notes(self):
        """Yield (section, track, note) for every melody note."""
        for sec in self.sections:
            for trk in sec.tracks:
                if trk.kind == "melody" and trk.notes:
                    for n in trk.notes:
                        yield sec, trk, n
