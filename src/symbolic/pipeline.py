"""One-call pipeline shared by the UI (app.py) and scripts:
   audio -> Song (extract + tempo + beat-grid quantize) -> annotate (key +
   fingerprint) -> reharmonize -> render. Thin orchestration over the engine modules.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from .model import (Song, Section, Track, Note, Pitch, Key, Provenance, Measured,
                    TempoEvent)
from . import extraction, analysis, transforms

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "spikes" / "_out"
OUT.mkdir(parents=True, exist_ok=True)
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aif", ".aiff")


def decode_to_wav(src) -> Path:
    src = Path(src)
    if src.suffix.lower() == ".wav":
        return src
    wav = OUT / (src.stem + ".wav")
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", str(wav)],
                   check=True, capture_output=True)
    return wav


def extract_to_song(wav_path, stem: str = "clip", ppq: int = 960, grid: int = 4,
                    bpm_override: float = 0) -> Song:
    """Extract melody (pYIN, auto-register, vibrato/onset-aware), detect tempo (or use
    bpm_override if > 0), and quantize onsets/durations to the beat grid. Raw timing ->
    measured layer."""
    wav = str(wav_path)
    notes = extraction.extract_pyin(wav)
    bpm = float(bpm_override) if bpm_override and bpm_override > 0 else \
        extraction.estimate_tempo(wav, notes=notes)[0]   # reuse notes (no 2nd pYIN)
    beat = 60.0 / bpm
    snap = max(1, ppq // grid)
    ordered = sorted(notes, key=lambda n: n["onset"])
    t0 = ordered[0]["onset"] if ordered else 0.0
    model_notes = []
    for i, n in enumerate(ordered):
        start = int(round((n["onset"] - t0) / beat * ppq / snap) * snap)
        dur = int(max(snap, round((n["offset"] - n["onset"]) / beat * ppq / snap) * snap))
        model_notes.append(Note(
            id=f"n{i}", startTick=start, durationTick=dur,
            pitch=Pitch.from_midi(n["midi"], prefer="sharp"),
            provenance=Provenance(source="extracted"),
            measured=Measured(onsetSec=round(n["onset"], 4),
                              durationSec=round(n["offset"] - n["onset"], 4),
                              f0Hz=n.get("f0Hz"), centsDev=n.get("centsDev"),
                              confidence=n.get("confidence"))))
    length = max((mn.startTick + mn.durationTick for mn in model_notes), default=ppq)
    return Song(
        id=f"song_{stem}", title=stem, key=Key(tonic="C", mode="ionian"),
        tempoMap=[TempoEvent(atTick=0, bpm=bpm)],
        sections=[Section(id="s", type="phrase", startTick=0, lengthTick=length,
                          tracks=[Track(id="lead", kind="melody", voiceId="lead",
                                        instrumentHint=stem, notes=model_notes)])])


def prep(annotated: Song, semitones: int = 0, mode: str = "keep") -> Song:
    """Apply optional refinement (transpose / parallel mode swap) and refresh the
    fingerprint, WITHOUT reharmonizing. Lets callers generate progression options."""
    work = annotated
    if semitones:
        work = transforms.transpose(work, int(semitones))
    if mode == "minor":
        work = transforms.mode_swap(work, "aeolian")
    elif mode == "major":
        work = transforms.mode_swap(work, "ionian")
    work.analysis = analysis.compute_fingerprint(work)
    return work


def arrange(annotated: Song, semitones: int = 0, mode: str = "keep",
            harmonic_beats: int = 2) -> Song:
    """Apply optional refinement (transpose / parallel mode swap), recompute the
    fingerprint, then reharmonize in the resulting key."""
    work = annotated
    if semitones:
        work = transforms.transpose(work, int(semitones))
    if mode == "minor":
        work = transforms.mode_swap(work, "aeolian")
    elif mode == "major":
        work = transforms.mode_swap(work, "ionian")
    work.analysis = analysis.compute_fingerprint(work)     # refresh after transforms
    return transforms.diatonic_reharmonize(work, harmonic_rhythm_ticks=harmonic_beats * work.ppq)


def _song_from_phrase(notes_list, ref: Song, stem: str, harmonic_beats: int) -> Song:
    """Build a reharmonized Song from a [{midi,durationTicks}] phrase, inheriting the
    reference Song's key + tempo."""
    ppq, t, mnotes = ref.ppq, 0, []
    for i, nd in enumerate(notes_list):
        mnotes.append(Note(id=f"n{i}", startTick=t, durationTick=int(nd["durationTicks"]),
                           pitch=Pitch.from_midi(int(nd["midi"]), prefer="sharp"),
                           provenance=Provenance(source="generated:variation")))
        t += int(nd["durationTicks"])
    song = Song(id=f"song_{stem}", title=stem, key=ref.key.model_copy(),
                tempoMap=[TempoEvent(atTick=0, bpm=ref.tempo_bpm())],
                sections=[Section(id="s", type="phrase", startTick=0, lengthTick=max(t, ppq),
                                  tracks=[Track(id="lead", kind="melody", voiceId="lead", notes=mnotes)])])
    song.analysis = analysis.compute_fingerprint(song)
    return transforms.diatonic_reharmonize(song, harmonic_rhythm_ticks=harmonic_beats * ppq)


def variations(work: Song, n: int = 3, model: str = "qwen2.5:7b", harmonic_beats: int = 2):
    """Idea expander: LLM melodic variations of `work` (in-key, motif-preserving) via the
    validate-and-repair loop, each returned as a reharmonized Song. Raises if Ollama is
    unreachable."""
    import json
    from . import variation as V
    from .theory import scale_pitch_classes
    if not V.ollama_alive():
        raise RuntimeError("Ollama not reachable - run `ollama serve` (or set OLLAMA_HOST).")
    mel = sorted((nt for sec in work.sections for t in sec.tracks
                  if t.kind == "melody" and t.notes for nt in t.notes), key=lambda x: x.startTick)
    if not mel:
        return []
    phrase = [{"midi": x.pitch.midi, "durationTicks": x.durationTick} for x in mel]
    midis = [x.pitch.midi for x in mel]
    ctx = {"scale_pcs": scale_pitch_classes(work.key.tonic, work.key.mode),
           "minMidi": min(midis) - 2, "maxMidi": max(midis) + 2,
           "motif_len": min(3, len(phrase)),
           "total_ticks": sum(p["durationTicks"] for p in phrase),
           "ppq": work.ppq, "key": f"{work.key.tonic} {work.key.mode}"}
    songs = []
    for i in range(n):
        try:
            text, _ = V.generate_ollama(model, V.build_messages(phrase, ctx),
                                        temperature=0.7 + 0.12 * i, seed=2000 + i)
            raw = json.loads(text).get("notes", [])
        except Exception:
            raw = []
        fixed = V.repair(raw, phrase, ctx)
        songs.append(_song_from_phrase(fixed, work, stem=f"var{i+1}", harmonic_beats=harmonic_beats))
    return songs
