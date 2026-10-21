"""Hand-built demo Song that exercises every layer of the schema:
two-layer fidelity (measured), symbols-primary harmony, locks (full + partial),
constraints with a motif, provenance, and integer-tick timing.

Reused by spikes B (round-trip), C (transforms), D (variation).
"""
from __future__ import annotations

from .model import (
    Song, Section, Track, Note, Chord, ChordRoot, Pitch, Measured, Lock,
    Provenance, Key, Meta, Constraints, Motif, PitchRange, TempoEvent,
    TimeSignatureEvent,
)

Q = 960  # ticks per quarter at ppq=960


def _note(nid, start_q, dur_q, step, alter, octave, *, measured=None, lock=None):
    return Note(
        id=nid, startTick=start_q * Q, durationTick=dur_q * Q,
        pitch=Pitch.from_spelling(step, alter, octave),
        velocity=80, lock=lock or Lock(),
        provenance=Provenance(source="extracted"), measured=measured,
    )


def demo_song() -> Song:
    melody = Track(
        id="trk_lead", kind="melody", voiceId="lead", role="lead",
        instrumentHint="voice",
        notes=[
            # "opening" motif C-D-E ; n1 fully locked to test lock-respecting transforms
            _note("n1", 0, 1, "C", 0, 4, lock=Lock(locked=True),
                  measured=Measured(onsetSec=0.012, durationSec=0.61, f0Hz=261.1,
                                    centsDev=-6.2, confidence=0.94)),
            _note("n2", 1, 1, "D", 0, 4,
                  measured=Measured(onsetSec=0.63, durationSec=0.60, f0Hz=293.0,
                                    centsDev=-3.1, confidence=0.91)),
            _note("n3", 2, 1, "E", 0, 4),
            _note("n4", 3, 1, "G", 0, 4),
            _note("n5", 4, 2, "E", 0, 4),         # half note
            _note("n6", 6, 1, "D", 0, 4),
            _note("n7", 7, 1, "C", 0, 4),
        ],
    )
    harmony = Track(
        id="trk_harm", kind="harmony", voiceId="harmony", role="chords",
        chords=[
            Chord(id="c1", startTick=0, durationTick=4 * Q, romanNumeral="I",
                  chordSymbol="C", root=ChordRoot(pitchClass=0, name="C"),
                  quality="major", inversion=0, chordPitchClasses=[0, 4, 7],
                  function="tonic", provenance=Provenance(source="generated:reharm-v1")),
            Chord(id="c2", startTick=4 * Q, durationTick=4 * Q, romanNumeral="V",
                  chordSymbol="G", root=ChordRoot(pitchClass=7, name="G"),
                  quality="major", inversion=0, chordPitchClasses=[7, 11, 2],
                  function="dominant", provenance=Provenance(source="generated:reharm-v1")),
        ],
    )
    section = Section(
        id="sec_verse1", type="verse", label="Verse 1",
        startTick=0, lengthTick=8 * Q,
        key=Key(tonic="C", mode="ionian"),
        constraints=Constraints(
            scalePitchClasses=[0, 2, 4, 5, 7, 9, 11],
            pitchRange=PitchRange(minMidi=60, maxMidi=79),
            motifs=[Motif(id="motif_a", noteIds=["n1", "n2", "n3"], label="opening")],
            styleHints="gentle, stepwise resolution",
        ),
        tracks=[melody, harmony],
    )
    return Song(
        id="song_demo_0001", title="Demo Hum",
        meta=Meta(ppq=Q, createdBy="fixtures.demo_song"),
        key=Key(tonic="C", mode="ionian", confidence=0.82),
        tempoMap=[TempoEvent(atTick=0, bpm=96.0)],
        timeSignatureMap=[TimeSignatureEvent(atTick=0, numerator=4, denominator=4)],
        sections=[section],
    )
