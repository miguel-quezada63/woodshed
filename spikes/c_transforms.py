"""Spike C - prove deterministic transforms are surgical, reversible, and
theory-correct. This is the real test of whether the data model supports
refinement.

Pass criteria (hardened per review):
  - interval-correct SPELLING: A4 above C is F#, d5 above C is Gb (same pitch,
    different spelling) - proves transpose is interval-based, not semitone+guess.
  - reversibility on the SYMBOLIC layer only (measured is nulled by design).
  - locks are honored (a frozen note is never altered).
  - reharmonization GUARANTEE: every strong-beat melody note is a chord tone,
    chords are diatonic, and the progression is not trivially all-I.
  - diffs are surgical: only intended fields change, ids stay stable.
  - every transform output still validates against song.schema.json.

Run: uv run python spikes/c_transforms.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jsonschema  # noqa: E402
from symbolic.model import Song  # noqa: E402
from symbolic.fixtures import demo_song  # noqa: E402
from symbolic import transforms as T  # noqa: E402

SCHEMA = json.loads((ROOT / "schema" / "song.schema.json").read_text(encoding="utf-8"))
_VALIDATOR = jsonschema.Draft202012Validator(SCHEMA)

results = []


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" - {detail}" if detail else ""))


def schema_ok(song: Song) -> bool:
    try:
        _VALIDATOR.validate(song.model_dump(mode="json", exclude_none=True))
        return True
    except jsonschema.ValidationError as e:
        print("    schema error:", e.message, list(e.path))
        return False


def melody(song):
    return [n for _, _, n in song.iter_melody_notes()]


def note_by_id(song, nid):
    return next(n for n in melody(song) if n.id == nid)


def signature(song):
    """Symbolic-only fingerprint (excludes measured/provenance/lock) for
    reversibility comparison."""
    return {
        "key": (song.key.tonic, song.key.mode),
        "notes": [(n.id, n.pitch.name, n.pitch.midi, n.startTick, n.durationTick)
                  for n in melody(song)],
        "sections": [(s.id, s.startTick, s.lengthTick, s.key.mode if s.key else None)
                     for s in song.sections],
    }


def test_transpose_spelling():
    print("\n[1] transpose - interval-correct spelling (A4 vs d5 from C)")
    base = demo_song()
    aug4 = T.transpose(base, "A4")     # C -> F#
    dim5 = T.transpose(base, "d5")     # C -> Gb (same pitch class, diff spelling)
    # n7 is a C4. Locked n1 is also C4 but frozen, so use n7.
    f_sharp = note_by_id(aug4, "n7").pitch
    g_flat = note_by_id(dim5, "n7").pitch
    check("A4 above C spells F#4", (f_sharp.step, f_sharp.alter) == ("F", 1), f_sharp.name)
    check("d5 above C spells Gb4", (g_flat.step, g_flat.alter) == ("G", -1), g_flat.name)
    check("...and they are the same pitch class", f_sharp.midi == g_flat.midi, f"{f_sharp.midi}")
    check("transpose output is schema-valid", schema_ok(aug4))


def test_transpose_reversible_locks_surgical():
    print("\n[2] transpose - reversibility (symbolic), locks, surgical diff, measured-null")
    base = demo_song()
    up = T.transpose(base, "P4")
    back = T.transpose(up, "-P4")
    check("P4 up then down restores symbolic signature", signature(back) == signature(base))

    # lock respected: n1 is fully locked -> never moves
    n1_up = note_by_id(up, "n1")
    check("locked n1 stays C4 through transpose", n1_up.pitch.name == "C4", n1_up.pitch.name)
    check("locked n1 keeps its measured data", n1_up.measured is not None
          and n1_up.measured.f0Hz == 261.1)

    # unlocked moved note: measured nulled, id stable
    n2_up = note_by_id(up, "n2")
    check("moved n2 had measured nulled", n2_up.measured is None)
    check("moved n2 kept stable id + provenance tag", n2_up.id == "n2"
          and n2_up.provenance.source == "transform:transpose")

    # surgical: timing + ids + structure unchanged (only pitch/key/measured move)
    same_timing = all(a[0] == b[0] and a[3] == b[3] and a[4] == b[4]
                      for a, b in zip(signature(up)["notes"], signature(base)["notes"]))
    check("timing + ids + note count unchanged (surgical)", same_timing
          and len(melody(up)) == len(melody(base)))


def test_mode_swap():
    print("\n[3] mode swap - parallel ionian<->aeolian, melody + chords")
    base = demo_song()
    minor = T.mode_swap(base, "aeolian")
    n3 = note_by_id(minor, "n3")   # E4 (deg 3) -> Eb4
    n2 = note_by_id(minor, "n2")   # D4 (deg 2) unchanged
    check("E4 (deg3) -> Eb4 in parallel minor", (n3.pitch.step, n3.pitch.alter) == ("E", -1), n3.pitch.name)
    check("D4 (deg2) unchanged", n2.pitch.name == "D4", n2.pitch.name)
    check("song key mode -> aeolian", minor.key.mode == "aeolian")

    harm = next(t for t in minor.sections[0].tracks if t.kind == "harmony")
    c1, c2 = harm.chords[0], harm.chords[1]
    check("I -> i (minor tonic), pcs [0,3,7]", c1.romanNumeral == "i"
          and c1.quality == "minor" and c1.chordPitchClasses == [0, 3, 7],
          f"{c1.romanNumeral}/{c1.chordPitchClasses}")
    check("V(root G,deg5) -> v, pcs [2,7,10]", c2.romanNumeral == "v"
          and c2.chordPitchClasses == [2, 7, 10], f"{c2.romanNumeral}/{c2.chordPitchClasses}")

    # reversibility of the swap on diatonic material
    backagain = T.mode_swap(minor, "ionian")
    check("ionian->aeolian->ionian restores signature", signature(backagain) == signature(base))
    check("mode swap output is schema-valid", schema_ok(minor))


def test_reharmonize():
    print("\n[4] diatonic reharmonization - strong-beat note is always a chord tone")
    base = demo_song()
    scale_pcs = set(base.sections[0].constraints.scalePitchClasses)
    re = T.diatonic_reharmonize(base, harmonic_rhythm_ticks=2 * base.ppq)

    harm = next(t for t in re.sections[0].tracks if t.kind == "harmony")
    mel = sorted(melody(re), key=lambda n: n.startTick)
    prog = []
    all_strong_ok, all_diatonic = True, True
    for ch in harm.chords:
        prog.append(ch.romanNumeral)
        # strong (slot-onset) melody note must be a chord tone
        sounding = [n for n in mel if n.startTick <= ch.startTick < n.startTick + n.durationTick]
        if sounding:
            pc = sounding[0].pitch.midi % 12
            if pc not in ch.chordPitchClasses:
                all_strong_ok = False
        if not set(ch.chordPitchClasses).issubset(scale_pcs):
            all_diatonic = False

    print(f"    progression: {' '.join(prog)}")
    check("every strong-beat melody note is a chord tone", all_strong_ok)
    check("every chord is diatonic to the key", all_diatonic)
    check("progression has variety (not all-I)", len(set(prog)) >= 2, f"{len(set(prog))} distinct")
    check("melody untouched by reharmonization", signature(re)["notes"] == signature(base)["notes"])
    check("reharmonization output is schema-valid", schema_ok(re))


def main():
    test_transpose_spelling()
    test_transpose_reversible_locks_surgical()
    test_mode_swap()
    test_reharmonize()
    ok = all(results)
    print(f"\nSPIKE C: {'PASS' if ok else 'FAIL'}  ({sum(results)}/{len(results)} checks)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
