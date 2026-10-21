"""Spike B - prove the data model is lossless and editable.

Two DISTINCT tests (kept separate so MIDI's known lossiness can't masquerade as
a schema bug):
  1. model -> JSON -> model         : the real contract test. MUST be 100%
     lossless, including measured / locks / provenance. Also schema-valid.
  2. model -> MIDI -> model         : musical-layer only (pitch + timing).
     MIDI cannot carry spelling, sections, roman numerals, measured, locks.
  3. model -> MusicXML -> reload     : spelling survives (the thing MIDI loses).

Run: uv run python spikes/b_roundtrip.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jsonschema  # noqa: E402

from symbolic.model import Song, Pitch, Note, Track, Section, Key, Lock, Provenance  # noqa: E402
from symbolic.fixtures import demo_song  # noqa: E402
from symbolic import io_render  # noqa: E402

OUT = ROOT / "spikes" / "_out"
OUT.mkdir(exist_ok=True)
SCHEMA = json.loads((ROOT / "schema" / "song.schema.json").read_text(encoding="utf-8"))

results = []


def check(name: str, ok: bool, detail: str = ""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def canon(song: Song) -> dict:
    """Canonical JSON form: drop absent/None optionals so it matches the schema
    (absent == null semantics) and is idempotent across round-trips."""
    return song.model_dump(mode="json", exclude_none=True)


def test_json_roundtrip():
    print("\n[1] model -> JSON -> model (contract; must be 100% lossless)")
    song = demo_song()
    d1 = canon(song)

    # schema conformance: the python model and the language-agnostic schema agree
    try:
        jsonschema.Draft202012Validator(SCHEMA).validate(d1)
        check("dump conforms to song.schema.json", True)
    except jsonschema.ValidationError as e:
        check("dump conforms to song.schema.json", False, f"{e.message} @ {list(e.path)}")
        return

    # round-trip
    song2 = Song.model_validate(d1)
    d2 = canon(song2)
    check("JSON->model->JSON deep-equal", d1 == d2)
    check("byte-identical (sorted)",
          json.dumps(d1, sort_keys=True) == json.dumps(d2, sort_keys=True))

    # guard against trivially-passing on a thin doc: the rich layers must be present
    n1 = d2["sections"][0]["tracks"][0]["notes"][0]
    check("measured layer preserved", n1.get("measured", {}).get("f0Hz") == 261.1, str(n1.get("measured")))
    check("lock preserved (n1 fully locked)", n1["lock"]["locked"] is True)
    check("provenance preserved", n1["provenance"]["source"] == "extracted")
    c1 = d2["sections"][0]["tracks"][1]["chords"][0]
    check("harmony symbol preserved", c1["romanNumeral"] == "I" and c1["chordPitchClasses"] == [0, 4, 7])
    check("constraints/motif preserved",
          d2["sections"][0]["constraints"]["motifs"][0]["noteIds"] == ["n1", "n2", "n3"])

    # an edit propagates: change one note's pitch, confirm it shows up
    edited = Song.model_validate(d1)
    edited.sections[0].tracks[0].notes[2].pitch = Pitch.from_spelling("F", 0, 4)
    check("edit propagates to export",
          io_render.melody_note_events(edited)[2]["midi"] == 65)


def test_midi_roundtrip():
    print("\n[2] model -> MIDI -> model (musical layer only; MIDI is lossy by design)")
    song = demo_song()
    expected = io_render.melody_note_events(song)
    midi_path = str(OUT / "demo.mid")
    io_render.export_midi(song, midi_path)
    got = io_render.import_midi_events(midi_path, song.ppq, song.tempo_bpm())

    same_len = len(expected) == len(got)
    check("note count preserved", same_len, f"{len(expected)} vs {len(got)}")
    if same_len:
        pitch_ok = all(e["midi"] == g["midi"] for e, g in zip(expected, got))
        time_ok = all(e["startTick"] == g["startTick"] and e["durationTick"] == g["durationTick"]
                      for e, g in zip(expected, got))
        check("pitches preserved (musical layer)", pitch_ok)
        check("onset+duration ticks preserved (musical layer)", time_ok,
              "" if time_ok else f"e0={expected[0]} g0={got[0]}")
    print("    (note: spelling, sections, roman numerals, measured, locks are NOT"
          " in MIDI - expected, that is why JSON is canonical)")


def test_musicxml_spelling():
    print("\n[3] model -> MusicXML -> reload (spelling survives; MIDI cannot do this)")
    # probe with enharmonically-ambiguous pitches: Bb4(70) vs A#, F#4(66) vs Gb
    probe = Song(
        id="song_probe", key=Key(tonic="F", mode="major"),
        sections=[Section(
            id="s", type="phrase", startTick=0, lengthTick=2 * 960,
            key=Key(tonic="F", mode="major"),
            tracks=[Track(id="t", kind="melody", notes=[
                Note(id="p1", startTick=0, durationTick=960, pitch=Pitch.from_spelling("B", -1, 4),
                     provenance=Provenance(source="user")),
                Note(id="p2", startTick=960, durationTick=960, pitch=Pitch.from_spelling("F", 1, 4),
                     provenance=Provenance(source="user")),
            ])],
        )],
    )
    xml_path = str(OUT / "probe.musicxml")
    io_render.export_musicxml(probe, xml_path)
    names = io_render.musicxml_pitch_names(xml_path)
    check("Bb4 stayed flat (not A#) through MusicXML", names[:1] == ["B-4"], str(names))
    check("F#4 stayed sharp through MusicXML", names[1:2] == ["F#4"], str(names))
    print(f"    MIDI would render these as just 70 and 66 - spelling unrecoverable.")


def main():
    test_json_roundtrip()
    test_midi_roundtrip()
    test_musicxml_spelling()
    ok = all(results)
    print(f"\nSPIKE B: {'PASS' if ok else 'FAIL'}  ({sum(results)}/{len(results)} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
