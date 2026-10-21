"""Monophonic melody extraction: f0 trackers -> note events -> key-invariant
intervals. Three backends compared in Spike A:
  - pYIN       (librosa, pure-python, frame f0)   -> segmented to notes here
  - torchcrepe (CREPE pytorch port, frame f0)     -> segmented to notes here
  - basic-pitch(quarantined TF venv, note events) -> via subprocess

Note dict shape: {"midi": int, "onset": float_sec, "offset": float_sec}.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def hz_to_midi(hz):
    hz = np.asarray(hz, dtype=float)
    out = np.full(hz.shape, np.nan)
    pos = np.isfinite(hz) & (hz > 0)
    out[pos] = 69.0 + 12.0 * np.log2(hz[pos] / 440.0)
    return out


def midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def frames_to_notes(times: np.ndarray, midi_float: np.ndarray, voiced: np.ndarray,
                    hop_sec: float, min_dur: float = 0.06, median_frames: int = 9,
                    change_semitones: float = 0.6, change_frames: int = 4,
                    f0_hz: np.ndarray = None, conf: np.ndarray = None,
                    min_confidence: float = 0.0, onset_frames=None) -> List[Dict]:
    """Segment a per-frame f0 contour into notes with VIBRATO-ROBUST HYSTERESIS:
    a note keeps its center pitch (running median of its frames) through wobble and
    only splits when the smoothed pitch moves >= change_semitones for change_frames
    consecutive frames. This stops vibrato from being chopped into an alternating-
    semitone trill. Each note's pitch is the MEDIAN of its frames (not a per-frame
    vote); per-note confidence / cents-deviation / f0 are recorded for the measured
    layer. Short blips dropped; brief unvoiced gaps bridged. Keep median_frames below
    the shortest expected note (in frames) to avoid merging genuinely fast notes."""
    from scipy.signal import medfilt

    n = len(midi_float)
    m = np.asarray(midi_float, dtype=float).copy()
    m[~voiced] = np.nan
    valid = np.isfinite(m)
    sm = m.copy()
    if median_frames > 1 and valid.sum() > 1:
        k = median_frames + 1 if median_frames % 2 == 0 else median_frames
        idx = np.arange(n)
        interp = np.interp(idx, idx[valid], m[valid])
        filt = medfilt(interp, kernel_size=k)
        sm[valid] = filt[valid]

    notes: List[Dict] = []
    start_i: Optional[int] = None
    frames: List[float] = []     # continuous pitches in the current note
    idxs: List[int] = []         # their frame indices
    pend: List = []              # (i, pitch) deviating candidates not yet committed
    gap = 0
    max_gap = 3
    onset_set = {int(f) for f in onset_frames} if onset_frames is not None else set()

    def flush(end_i: int):
        if start_i is None or not frames:
            return
        onset = float(times[start_i])
        offset = float(times[min(end_i, n - 1)]) + hop_sec
        if offset - onset < min_dur:
            return
        pitch = int(round(float(np.median(frames))))
        note = {"midi": pitch, "onset": onset, "offset": offset}
        if conf is not None:
            cv = np.asarray(conf)[idxs]
            cv = cv[np.isfinite(cv)]
            if cv.size:
                note["confidence"] = round(float(np.mean(cv)), 4)
        if min_confidence > 0 and note.get("confidence", 1.0) < min_confidence:
            return  # low-confidence note (e.g. breath/glottal edge artifact) - drop
        if f0_hz is not None:
            fv = np.asarray(f0_hz)[idxs]
            fv = fv[np.isfinite(fv) & (fv > 0)]
            if fv.size:
                note["f0Hz"] = round(float(np.median(fv)), 3)
                note["centsDev"] = round(float(np.median(1200 * np.log2(fv / midi_to_hz(pitch)))), 1)
        notes.append(note)

    for i in range(n):
        if not valid[i]:
            gap += 1
            if start_i is not None and gap > max_gap:
                flush(i - gap + 1)
                start_i, frames, idxs, pend = None, [], [], []
            continue
        gap = 0
        p = float(sm[i])
        if start_i is None:
            start_i, frames, idxs, pend = i, [p], [i], []
            continue
        if onset_set and i in onset_set and (i - start_i) * hop_sec >= min_dur:
            flush(i)                                             # re-attack of a HELD note -> split
            start_i, frames, idxs, pend = i, [p], [i], []        #  (vibrato still merged within a note)
            continue
        if abs(p - float(np.median(frames))) < change_semitones:
            frames.append(p); idxs.append(i); pend = []          # within wobble -> same note
        else:
            pend.append((i, p))                                  # deviating candidate
            if len(pend) >= change_frames:                       # persisted -> real change
                new_start = pend[0][0]
                flush(new_start)
                start_i, frames, idxs = new_start, [pp for _, pp in pend], [ii for ii, _ in pend]
                pend = []
    if start_i is not None:
        flush(n - 1)
    return notes


def extract_pyin(wav_path: str, sr: int = 22050, hop: int = 256,
                 fmin: float = None, fmax: float = None,
                 auto_register: bool = True, min_confidence: float = 0.12,
                 onset_ratio: float = 3.0, frame_length: int = 2048,
                 pyin=None) -> List[Dict]:
    """pYIN melody extraction. By default it AUTO-DETECTS the register: a coarse
    wide-band pass finds the median voiced pitch, then fmin/fmax are re-bracketed
    around it (one octave each side). This prevents the classic failure on real
    high whistles, where a wide fmin lets pYIN lock onto octave-down subharmonics
    at note boundaries. Pass explicit fmin/fmax to override. `pyin` swaps in a
    drop-in replacement for librosa.pyin (e.g. a chunked multi-process one)."""
    import librosa
    pyin = pyin or librosa.pyin
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    lo_c2, hi_c7 = float(librosa.note_to_hz("C2")), float(librosa.note_to_hz("C7"))  # C2 floor: avoids sub-octave artifacts on clean voice/instrument input

    if (fmin is None or fmax is None) and auto_register:
        # coarse grid (half-semitone bins, 4x hop): only the median is needed, and
        # pYIN's cost grows with bins^2 - this is ~40x cheaper than a full-res pass
        f0c, _, _ = pyin(y, sr=sr, fmin=lo_c2, fmax=hi_c7, hop_length=hop * 4, resolution=0.5)
        voiced_hz = f0c[np.isfinite(f0c)]
        if voiced_hz.size:
            med = float(np.median(voiced_hz))
            fmin = fmin or max(lo_c2, med / 2.0)
            fmax = fmax or min(hi_c7, med * 2.0)
    fmin = fmin or lo_c2
    fmax = fmax or hi_c7

    f0, vflag, vprob = pyin(y, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop,
                            frame_length=frame_length)
    times = librosa.times_like(f0, sr=sr, hop_length=hop)
    midi = hz_to_midi(f0)
    voiced = np.isfinite(midi) & (vflag.astype(bool))
    # Only STRONG re-attacks split a held note (rhythmic re-articulation). This keeps
    # sustained tones (a vibrato hum, a clean synth note) from being chopped up by the
    # weak amplitude bumps librosa would otherwise flag as onsets.
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onsets = np.asarray(librosa.onset.onset_detect(
        onset_envelope=env, sr=sr, hop_length=hop, units="frames", backtrack=False), dtype=int)
    if onsets.size:
        active = env[env > env.max() * 0.05]
        floor = onset_ratio * float(np.median(active)) if active.size else 0.0
        onsets = onsets[env[onsets] >= floor]
    return frames_to_notes(times, midi, voiced, hop_sec=hop / sr, f0_hz=f0, conf=vprob,
                           min_confidence=min_confidence, onset_frames=onsets)


def _grid_fit_error(onsets, bpm, ppq=960, grid=4):
    """Mean distance (0=on-grid .. 0.5=between) of note onsets from a `grid`-per-beat
    grid at `bpm`. Lets us resolve tempo octave (2x/half) ambiguity by data."""
    if len(onsets) < 2:
        return 1.0
    beat = 60.0 / bpm
    snap = ppq / grid
    t0 = onsets[0]
    errs = [abs((o - t0) / beat * ppq - round((o - t0) / beat * ppq / snap) * snap) for o in onsets]
    return float(np.mean(errs)) / snap


def estimate_tempo(wav_path: str, sr: int = 22050, notes=None):
    """Estimate tempo (BPM) and beat times (sec). Monophonic legato hum/whistle has
    weak onsets, so beat-tracking is only a seed: we generate octave variants plus an
    inter-onset-interval estimate and pick the tempo whose grid best fits the actual
    note onsets (gently preferring a musical 70-140 range). Fixes the common 2x error.
    Pass already-extracted `notes` to avoid a redundant pYIN pass."""
    import librosa
    pyin = pyin or librosa.pyin
    y, sr = librosa.load(wav_path, sr=sr, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    base = float(np.atleast_1d(tempo)[0])

    if notes is None:
        notes = extract_pyin(wav_path, sr=sr)
    onsets = sorted(n["onset"] for n in notes)

    cands = {round(b, 1) for b in (base, base / 2, base * 2) if 55 <= b <= 190}
    iois = np.diff(onsets) if len(onsets) > 2 else np.array([])
    iois = iois[iois > 0.08]
    if iois.size:
        ib = 60.0 / float(np.median(iois))
        while ib < 55:
            ib *= 2
        while ib > 190:
            ib /= 2
        cands |= {round(b, 1) for b in (ib, ib * 2) if 55 <= b <= 190}
    if not cands:
        cands = {round(base, 1) if 30 < base else 100.0}

    def score(b):
        return _grid_fit_error(onsets, b) + (0.0 if 70 <= b <= 140 else 0.03)

    best = min(cands, key=score)
    return round(best, 1), librosa.frames_to_time(beats, sr=sr)


def extract_torchcrepe(wav_path: str, model: str = "full", hop: int = 160) -> List[Dict]:
    import librosa
    import torch
    import torchcrepe
    sr = 16000
    y, _ = librosa.load(wav_path, sr=sr, mono=True)
    tens = torch.from_numpy(y.astype(np.float32))[None]
    pitch, periodicity = torchcrepe.predict(
        tens, sr, hop_length=hop, fmin=50.0, fmax=2006.0, model=model,
        return_periodicity=True, batch_size=1024, device="cpu")
    periodicity = torchcrepe.filter.median(periodicity, 3)
    pitch = torchcrepe.filter.mean(pitch, 3)
    hz = pitch.squeeze(0).cpu().numpy()
    per = periodicity.squeeze(0).cpu().numpy()
    midi = hz_to_midi(hz)
    voiced = np.isfinite(midi) & (per > 0.5)
    times = np.arange(len(hz)) * hop / sr
    return frames_to_notes(times, midi, voiced, hop_sec=hop / sr, f0_hz=hz, conf=per)


def extract_basic_pitch_batch(wav_paths: List[str], venv: str = ".venv-bp") -> Dict[str, List[Dict]]:
    """Run basic-pitch in its quarantined venv via a single subprocess (amortizes
    the TensorFlow cold start). Returns {wav_path: notes}; raises RuntimeError with
    captured stderr on failure."""
    py = ROOT / venv / "Scripts" / "python.exe"
    runner = ROOT / "tools" / "bp_runner.py"
    out_path = Path(tempfile.mktemp(suffix="_bp.json"))
    env = dict(os.environ, TF_CPP_MIN_LOG_LEVEL="3", TF_ENABLE_ONEDNN_OPTS="0")
    proc = subprocess.run([str(py), str(runner), str(out_path), *map(str, wav_paths)],
                          capture_output=True, text=True, timeout=600, env=env)
    if proc.returncode != 0 or not out_path.exists():
        raise RuntimeError(proc.stderr.strip()[-1500:] or "basic-pitch failed")
    data = json.loads(out_path.read_text(encoding="utf-8"))
    out_path.unlink(missing_ok=True)
    return data


def extract_basic_pitch(wav_path: str, venv: str = ".venv-bp") -> List[Dict]:
    return extract_basic_pitch_batch([wav_path], venv)[str(wav_path)]


def notes_to_intervals(notes: List[Dict]) -> List[int]:
    """Key-invariant melodic interval sequence (semitone deltas), by onset."""
    ordered = sorted(notes, key=lambda n: n["onset"])
    return [ordered[i + 1]["midi"] - ordered[i]["midi"] for i in range(len(ordered) - 1)]
