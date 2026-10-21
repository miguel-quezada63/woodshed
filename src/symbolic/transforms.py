"""Tier-1 deterministic theory transforms. Each is a PURE function Song -> Song:
deep-copies, never mutates the input, keeps element ids STABLE (so diffs are
surgical), respects locks (frozen elements untouched), and nulls a note's
`measured` block whenever its pitch/timing changes (the performance no longer
corresponds - this is why reversibility is scoped to symbolic fields only).
"""
from __future__ import annotations

from typing import List, Optional

import music21 as m21

from .model import Song, Chord, Track, ChordRoot, Provenance, Note
from .theory import (
    make_scale, diatonic_triads, scale_pitch_classes, pitch_to_m21, m21_to_pitch,
    ours_name_to_m21, chord_symbol, Triad,
)


# ---------------------------------------------------------------- helpers
def _parse_interval(spec) -> m21.interval.Interval:
    """Accept a named interval 'P4'/'M3'/'-P4' (spelling-correct), or an int number
    of semitones (music21 picks a sensible spelling). Named intervals keep enharmonic
    intent; ints are convenient for a UI slider."""
    if isinstance(spec, int):
        return m21.interval.Interval(spec)
    if spec.startswith("-"):
        return m21.interval.Interval(spec[1:]).reverse()
    return m21.interval.Interval(spec)


def _touch_pitch(note: Note, transform: str) -> None:
    note.measured = None  # stale: performance was not at the new pitch/time
    note.provenance = Provenance(source=f"transform:{transform}")


def _locked(*locks, dimension: str) -> bool:
    return any(lk.freezes(dimension) for lk in locks)


def _transpose_key(key, iv: m21.interval.Interval) -> None:
    p = m21.pitch.Pitch(ours_name_to_m21(key.tonic) + "4").transpose(iv)
    key.tonic = p.name.replace("-", "b")


def _transpose_chord(ch: Chord, iv: m21.interval.Interval) -> None:
    semis = iv.semitones
    if ch.root:
        rp = m21.pitch.Pitch(ours_name_to_m21(ch.root.name) + "4").transpose(iv)
        ch.root = ChordRoot(pitchClass=rp.pitchClass, name=rp.name.replace("-", "b"))
        if ch.chordSymbol and ch.quality:
            ch.chordSymbol = chord_symbol(ch.root.name, ch.quality)
    if ch.chordPitchClasses:
        ch.chordPitchClasses = sorted({(pc + semis) % 12 for pc in ch.chordPitchClasses})
    if ch.voicing:
        ch.voicing = [v + semis for v in ch.voicing]
    ch.provenance = Provenance(source="transform:transpose")


# ---------------------------------------------------------------- transpose
def transpose(song: Song, interval: str) -> Song:
    """Transpose the whole song by a named interval (e.g. 'P4', '-m3').
    romanNumerals are key-relative so they stay valid; the key is transposed too."""
    iv = _parse_interval(interval)
    out = song.model_copy(deep=True)
    _transpose_key(out.key, iv)
    for sec in out.sections:
        if sec.key:
            _transpose_key(sec.key, iv)
        for trk in sec.tracks:
            if trk.kind == "melody" and trk.notes:
                for n in trk.notes:
                    if _locked(n.lock, trk.lock, sec.lock, dimension="pitch"):
                        continue
                    n.pitch = m21_to_pitch(pitch_to_m21(n.pitch).transpose(iv))
                    _touch_pitch(n, "transpose")
            elif trk.kind == "harmony" and trk.chords:
                for ch in trk.chords:
                    if _locked(ch.lock, trk.lock, sec.lock, dimension="harmony"):
                        continue
                    _transpose_chord(ch, iv)
    return out


# ---------------------------------------------------------------- mode swap
def mode_swap(song: Song, target_mode: str) -> Song:
    """Parallel mode swap (same tonic, new mode), e.g. ionian<->aeolian.
    Melody notes are remapped degree-for-degree through the new scale; chords are
    rebuilt from their root's scale degree in the new mode. Non-diatonic notes are
    left untouched. Invertible for diatonic material (tested per-transform)."""
    out = song.model_copy(deep=True)
    for sec in out.sections:
        src_key = sec.key or out.key
        tonic, src_mode = src_key.tonic, src_key.mode
        src_scale = make_scale(tonic, src_mode)
        tgt_scale = make_scale(tonic, target_mode)
        tgt_triads = diatonic_triads(tonic, target_mode)

        for trk in sec.tracks:
            if trk.kind == "melody" and trk.notes:
                for n in trk.notes:
                    if _locked(n.lock, trk.lock, sec.lock, dimension="pitch"):
                        continue
                    mp = pitch_to_m21(n.pitch)
                    deg = src_scale.getScaleDegreeFromPitch(mp, comparisonAttribute="pitchClass")
                    if not deg:
                        continue  # chromatic / non-diatonic: leave as-is
                    new_p = tgt_scale.pitchFromDegree(deg)
                    new_p.octave = mp.octave
                    n.pitch = m21_to_pitch(new_p)
                    _touch_pitch(n, "modeSwap")
            elif trk.kind == "harmony" and trk.chords:
                for ch in trk.chords:
                    if _locked(ch.lock, trk.lock, sec.lock, dimension="harmony") or not ch.root:
                        continue
                    rp = m21.pitch.Pitch(ours_name_to_m21(ch.root.name) + "4")
                    deg = src_scale.getScaleDegreeFromPitch(rp, comparisonAttribute="pitchClass")
                    if not deg:
                        continue
                    tri = tgt_triads[deg]
                    ch.romanNumeral = tri.roman
                    ch.quality = tri.quality
                    ch.function = tri.function
                    ch.root = tri.root
                    ch.chordPitchClasses = tri.pitch_classes
                    ch.chordSymbol = chord_symbol(tri.root.name, tri.quality)
                    ch.provenance = Provenance(source="transform:modeSwap")
        if sec.key:
            sec.key.mode = target_mode
    out.key.mode = target_mode
    return out


# ---------------------------------------------------------------- reharmonize
def _sounding_note(melody: List[Note], tick: int) -> Optional[Note]:
    sounding = [n for n in melody if n.startTick <= tick < n.startTick + n.durationTick]
    if sounding:
        return sounding[0]
    later = [n for n in melody if n.startTick >= tick]
    return later[0] if later else (melody[-1] if melody else None)


def _choose_triad(triads, scale_pcs, strong: Optional[Note], prev_roman: Optional[str],
                  rank: int = 0) -> Triad:
    """Pick the rank-th best diatonic triad that contains the strong-beat note
    (rank 0 = best). Every candidate contains the note, so any rank preserves the
    strong-beat-chord-tone guarantee; varying rank yields distinct valid progressions."""
    if strong is None:
        return triads[1]
    pc = strong.pitch.midi % 12
    candidates = [t for t in triads.values() if pc in t.pitch_classes]
    if not candidates:
        # non-chord strong tone (NCT): accented appoggiatura, resolve over tonic.
        return triads[1]
    scored = []
    for t in candidates:
        root_pc = t.root.pitchClass
        others = [x for x in t.pitch_classes if x != root_pc]
        third_pc = next((x for x in others if (x - root_pc) % 12 in (3, 4)), None)
        w = 3 if pc == root_pc else (2 if pc == third_pc else 1)        # prefer note as root
        score = w + (1 if t.degree in (1, 4, 5) else 0)                 # prefer primary triads
        if t.roman == prev_roman:
            score -= 2                                                  # discourage repeats
        scored.append((score, t.degree, t))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return scored[min(rank, len(scored) - 1)][2]


def diatonic_reharmonize(song: Song, harmonic_rhythm_ticks: Optional[int] = None,
                         rank: int = 0) -> Song:
    """Replace the harmony with diatonic triads, one per harmonic-rhythm slot.
    GUARANTEE: the strong-beat (slot-onset) melody note is a chord tone of the
    chosen chord; non-strong-beat notes may be non-chord tones (passing/neighbor).
    """
    out = song.model_copy(deep=True)
    for sec in out.sections:
        if _locked(sec.lock, dimension="harmony"):
            continue
        key = sec.key or out.key
        triads = diatonic_triads(key.tonic, key.mode)
        scale_pcs = set(scale_pitch_classes(key.tonic, key.mode))
        hr = (harmonic_rhythm_ticks
              or (sec.constraints.harmonicRhythmTicks if sec.constraints and sec.constraints.harmonicRhythmTicks else None)
              or 2 * out.ppq)

        melody = sorted(
            (n for trk in sec.tracks if trk.kind == "melody" and trk.notes for n in trk.notes),
            key=lambda n: n.startTick)

        new_chords, prev = [], None
        for s in range(0, sec.lengthTick, hr):
            strong = _sounding_note(melody, s)
            tri = _choose_triad(triads, scale_pcs, strong, prev, rank=rank)
            new_chords.append(Chord(
                id=f"ch_reharm_{s}", startTick=s, durationTick=min(hr, sec.lengthTick - s),
                romanNumeral=tri.roman, chordSymbol=chord_symbol(tri.root.name, tri.quality),
                root=tri.root, quality=tri.quality, inversion=0,
                chordPitchClasses=tri.pitch_classes, function=tri.function,
                provenance=Provenance(source="generated:reharm-diatonic")))
            prev = tri.roman

        harm = next((t for t in sec.tracks if t.kind == "harmony"), None)
        if harm is not None:
            if _locked(harm.lock, dimension="harmony"):
                continue
            harm.chords = new_chords
        else:
            sec.tracks.append(Track(id=f"trk_harm_{sec.id}", kind="harmony",
                                    voiceId="harmony", role="chords", chords=new_chords))
    return out


def progression_of(song: Song) -> List[str]:
    """The roman-numeral progression of the first harmony track (for display)."""
    for sec in song.sections:
        for trk in sec.tracks:
            if trk.kind == "harmony" and trk.chords:
                return [c.romanNumeral or "?" for c in trk.chords]
    return []


def reharmonize_options(song: Song, n: int = 3,
                        harmonic_rhythm_ticks: Optional[int] = None) -> List[Song]:
    """Return up to n DISTINCT diatonic reharmonizations (the chord co-pilot). Each
    satisfies the strong-beat-chord-tone guarantee; they differ by choosing lower-
    ranked (but still valid) chords per slot."""
    out, seen = [], set()
    for rank in range(n + 4):
        s = diatonic_reharmonize(song, harmonic_rhythm_ticks, rank=rank)
        key = tuple(progression_of(s))
        if key and key not in seen:
            seen.add(key)
            out.append(s)
        if len(out) >= n:
            break
    return out or [diatonic_reharmonize(song, harmonic_rhythm_ticks)]
