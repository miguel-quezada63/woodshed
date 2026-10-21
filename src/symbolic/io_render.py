"""Render/interchange IO. MIDI and MusicXML are EXPORT/IMPORT targets, never the
source of truth. MIDI is intentionally lossy (no spelling, sections, roman
numerals, measured, locks); MusicXML preserves spelling. The JSON document is
canonical.

Timing assumes a single constant tempo (tempoMap[0]); multi-tempo is future work.
"""
from __future__ import annotations

from typing import List, Dict

from .model import Song


def _tick_to_sec(abs_tick: int, ppq: int, bpm: float) -> float:
    return (abs_tick / ppq) * (60.0 / bpm)


def _sec_to_tick(sec: float, ppq: int, bpm: float) -> int:
    return round(sec * (bpm / 60.0) * ppq)


def melody_note_events(song: Song) -> List[Dict]:
    """Flat, absolute-tick view of all melody notes (the 'musical layer' that
    MIDI can carry). Sorted by onset for stable comparison."""
    out = []
    for sec, trk, n in song.iter_melody_notes():
        out.append({
            "midi": n.pitch.midi,
            "startTick": sec.startTick + n.startTick,
            "durationTick": n.durationTick,
            "velocity": n.velocity,
        })
    out.sort(key=lambda e: (e["startTick"], e["midi"]))
    return out


# General-MIDI program hints per voice (FL re-assigns instruments anyway; these
# just make the tracks land on sensible defaults).
_GM = {"lead": 80, "bass": 33, "inner": 4, "harmony": 48, "chords": 48}


def export_midi(song: Song, path: str, add_bass_from_chords: bool = True) -> str:
    """Clean multi-track MIDI for a DAW (FL Studio): one track per melody voice, a
    block-chord track, and a bass track from the chord roots - grid-aligned (symbolic
    ticks, not raw performance timing) at the detected tempo. Drag-and-drop ready."""
    import pretty_midi
    ppq, bpm = song.ppq, song.tempo_bpm()
    pm = pretty_midi.PrettyMIDI(resolution=ppq, initial_tempo=bpm)

    def sec_at(tick):
        return _tick_to_sec(tick, ppq, bpm)

    # one MIDI instrument per melody-kind track (lead / inner / bass / ...)
    for sec in song.sections:
        for trk in sec.tracks:
            if trk.kind != "melody" or not trk.notes:
                continue
            name = trk.voiceId or trk.role or "Melody"
            inst = pretty_midi.Instrument(program=_GM.get(trk.voiceId or "", 80), name=name)
            for n in trk.notes:
                inst.notes.append(pretty_midi.Note(
                    velocity=n.velocity, pitch=n.pitch.midi,
                    start=sec_at(sec.startTick + n.startTick),
                    end=sec_at(sec.startTick + n.startTick + n.durationTick)))
            pm.instruments.append(inst)

    # harmony -> block-chord track (+ optional bass track from roots)
    chords = pretty_midi.Instrument(program=_GM["chords"], name="Chords")
    bass = pretty_midi.Instrument(program=_GM["bass"], name="Bass")
    for sec in song.sections:
        for trk in sec.tracks:
            if trk.kind != "harmony" or not trk.chords:
                continue
            for ch in trk.chords:
                s, e = sec_at(sec.startTick + ch.startTick), sec_at(sec.startTick + ch.startTick + ch.durationTick)
                for m in (ch.voicing or [60 + pc for pc in (ch.chordPitchClasses or [])]):
                    chords.notes.append(pretty_midi.Note(velocity=70, pitch=m, start=s, end=e))
                if add_bass_from_chords:
                    root = ch.root.pitchClass if ch.root else (ch.chordPitchClasses or [0])[0]
                    bass.notes.append(pretty_midi.Note(velocity=84, pitch=36 + root, start=s, end=e))
    if chords.notes:
        pm.instruments.append(chords)
    if bass.notes:
        pm.instruments.append(bass)
    pm.write(path)
    return path


def import_midi_events(path: str, ppq: int, bpm: float, instrument_name: str = "lead") -> List[Dict]:
    """Read MIDI back into absolute-tick note events (musical layer only)."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(path)
    out = []
    for inst in pm.instruments:
        if inst.name and inst.name != instrument_name:
            continue
        for n in inst.notes:
            start_tick = _sec_to_tick(n.start, ppq, bpm)
            dur_tick = _sec_to_tick(n.end - n.start, ppq, bpm)
            out.append({"midi": n.pitch, "startTick": start_tick,
                        "durationTick": dur_tick, "velocity": n.velocity})
    out.sort(key=lambda e: (e["startTick"], e["midi"]))
    return out


def render_sine(song: Song, path: str, sr: int = 22050, use_measured: bool = True,
                a4: float = 440.0) -> str:
    """Disposable audio render of the melody as sine tones (whistle ~ sine, so this
    is a fair ear test). use_measured=True plays the RAW extracted timing
    (measured.onsetSec/durationSec) - faithful to what the tracker heard;
    False plays the quantized symbolic grid (the cleaned/editable version)."""
    import numpy as np
    import soundfile as sf

    ppq, bpm = song.ppq, song.tempo_bpm()
    events = []  # (onset_sec, dur_sec, midi)
    for sec, trk, n in song.iter_melody_notes():
        if use_measured and n.measured and n.measured.onsetSec is not None:
            onset, dur = n.measured.onsetSec, (n.measured.durationSec or 0.2)
        else:
            onset = _tick_to_sec(sec.startTick + n.startTick, ppq, bpm)
            dur = _tick_to_sec(n.durationTick, ppq, bpm)
        events.append((onset, max(dur, 0.03), n.pitch.midi))

    if not events:
        raise ValueError("no melody notes to render")
    total = max(o + d for o, d, _ in events) + 0.3
    audio = np.zeros(int(total * sr) + 1, dtype=np.float32)
    for onset, dur, midi in events:
        f = a4 * 2 ** ((midi - 69) / 12)
        n = int(dur * sr)
        t = np.arange(n) / sr
        env = np.minimum(1.0, np.minimum(t / 0.01, (dur - t) / 0.02))  # de-click fades
        sig = (0.5 * np.sin(2 * np.pi * f * t) * env).astype(np.float32)
        i = int(onset * sr)
        audio[i:i + n] += sig
    peak = float(np.max(np.abs(audio))) or 1.0
    sf.write(path, (0.9 * audio / peak).astype(np.float32), sr)
    return path


def _add_sine(audio, sr, freq, onset_sec, dur_sec, amp):
    import numpy as np
    n = int(max(dur_sec, 0.03) * sr)
    if n <= 0:
        return
    t = np.arange(n) / sr
    env = np.minimum(1.0, np.minimum(t / 0.01, (max(dur_sec, 0.03) - t) / 0.03))
    sig = (amp * np.sin(2 * np.pi * freq * t) * env).astype("float32")
    i = int(onset_sec * sr)
    end = min(len(audio), i + n)
    audio[i:end] += sig[: end - i]


def render_arrangement(song: Song, path: str, sr: int = 22050,
                       use_measured_melody: bool = True, a4: float = 440.0) -> str:
    """Render melody (sine) + harmony chords (soft block sines) to one wav. The
    reharmonization made audible. Melody can follow raw extracted timing; chords
    follow the symbolic tick grid."""
    import numpy as np

    ppq, bpm = song.ppq, song.tempo_bpm()
    mel, chords = [], []
    for sec, trk, n in song.iter_melody_notes():
        if use_measured_melody and n.measured and n.measured.onsetSec is not None:
            onset, dur = n.measured.onsetSec, (n.measured.durationSec or 0.2)
        else:
            onset = _tick_to_sec(sec.startTick + n.startTick, ppq, bpm)
            dur = _tick_to_sec(n.durationTick, ppq, bpm)
        mel.append((onset, max(dur, 0.03), n.pitch.midi))
    for sec in song.sections:
        for trk in sec.tracks:
            if trk.kind == "harmony" and trk.chords:
                for ch in trk.chords:
                    onset = _tick_to_sec(sec.startTick + ch.startTick, ppq, bpm)
                    dur = _tick_to_sec(ch.durationTick, ppq, bpm)
                    pcs = ch.voicing or [48 + pc for pc in (ch.chordPitchClasses or [])]
                    for m in pcs:
                        chords.append((onset, dur, m if m >= 36 else 48 + (m % 12)))

    total = max([o + d for o, d, _ in mel + chords], default=1.0) + 0.3
    audio = np.zeros(int(total * sr) + 1, dtype=np.float32)
    for onset, dur, midi in chords:                      # chords first (background)
        _add_sine(audio, sr, a4 * 2 ** ((midi - 69) / 12), onset, dur, amp=0.16)
    for onset, dur, midi in mel:                         # melody on top
        _add_sine(audio, sr, a4 * 2 ** ((midi - 69) / 12), onset, dur, amp=0.55)
    peak = float(np.max(np.abs(audio))) or 1.0
    import soundfile as sf
    sf.write(path, (0.9 * audio / peak).astype(np.float32), sr)
    return path


def export_musicxml(song: Song, path: str) -> str:
    """Export via music21, preserving pitch SPELLING (the thing MIDI loses)."""
    import music21 as m21
    score = m21.stream.Score()
    ppq = song.ppq
    for sec in song.sections:
        for trk in sec.tracks:
            if trk.kind != "melody" or not trk.notes:
                continue
            part = m21.stream.Part()
            part.id = trk.id
            if sec.key:
                part.append(m21.key.Key(sec.key.tonic,
                                        "major" if sec.key.mode in ("major", "ionian") else "minor"))
            for n in trk.notes:
                p = m21.pitch.Pitch(step=n.pitch.step, octave=n.pitch.octave)
                if n.pitch.alter:
                    p.accidental = m21.pitch.Accidental(n.pitch.alter)
                m21n = m21.note.Note(p, quarterLength=n.durationTick / ppq)
                part.insert((sec.startTick + n.startTick) / ppq, m21n)
            score.insert(0, part)
    score.write("musicxml", fp=path)
    return path


def musicxml_pitch_names(path: str) -> List[str]:
    """Reload a MusicXML file and return spelled pitch names in onset order."""
    import music21 as m21
    s = m21.converter.parse(path)
    notes = sorted(s.recurse().notes, key=lambda x: x.getOffsetInHierarchy(s))
    return [n.pitch.nameWithOctave for n in notes if n.isNote]
