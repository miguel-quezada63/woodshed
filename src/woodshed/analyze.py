"""Song-level analysis from AUDIO (not from a transcribed melody): key + Camelot,
tempo + beat grid.

Key: Krumhansl-Kessler profile correlation against the time-summed CQT chroma of
the pitched stems (drums excluded), all 24 keys ranked. Beats: librosa's beat
tracker on the drum stem when there is one (cleanest pulse), else the mix.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

# Krumhansl-Kessler key profiles
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# conventional spellings for key names
MAJOR_NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
MINOR_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
# keys whose signatures use flats (spell transcribed notes with flats in these)
FLAT_MAJOR = {5, 10, 3, 8, 1}          # F Bb Eb Ab Db
FLAT_MINOR = {2, 7, 0, 5, 10, 3}       # Dm Gm Cm Fm Bbm Ebm


@dataclass
class KeyInfo:
    tonic_pc: int
    mode: str                     # "major" | "minor"
    confidence: float             # correlation of the winning profile
    ranked: List[tuple] = field(default_factory=list)   # [(name, r), ...]

    @property
    def tonic(self) -> str:
        return (MAJOR_NAMES if self.mode == "major" else MINOR_NAMES)[self.tonic_pc]

    @property
    def name(self) -> str:
        return key_name(self.tonic_pc, self.mode)

    @property
    def camelot(self) -> str:
        return camelot(self.tonic_pc, self.mode)

    @property
    def prefers_flats(self) -> bool:
        return self.tonic_pc in (FLAT_MAJOR if self.mode == "major" else FLAT_MINOR)

    @property
    def relative(self) -> str:
        if self.mode == "major":
            return key_name((self.tonic_pc + 9) % 12, "minor")
        return key_name((self.tonic_pc + 3) % 12, "major")


def key_name(pc: int, mode: str) -> str:
    return f"{(MAJOR_NAMES if mode == 'major' else MINOR_NAMES)[pc % 12]} {mode}"


def camelot(pc: int, mode: str) -> str:
    """Camelot wheel code: C major = 8B, A minor = 8A; +1 per fifth up."""
    major_pc = pc if mode == "major" else (pc + 3) % 12
    num = (7 * major_pc + 8) % 12 or 12
    return f"{num}{'B' if mode == 'major' else 'A'}"


def camelot_neighbors(code: str) -> List[str]:
    """Harmonically compatible codes (same, +/-1, relative major/minor)."""
    num, letter = int(code[:-1]), code[-1]
    up, down = num % 12 + 1, (num - 2) % 12 + 1
    return [f"{up}{letter}", f"{down}{letter}", f"{num}{'A' if letter == 'B' else 'B'}"]


def detect_key(wavs: Sequence[Path], sr: int = 22050) -> KeyInfo:
    import librosa
    chroma_sum = np.zeros(12)
    for w in wavs:
        y, _ = librosa.load(str(w), sr=sr, mono=True)
        if not np.any(y):
            continue
        c = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=2048)
        chroma_sum += c.sum(axis=1)
    scores = []
    for pc in range(12):
        for mode, prof in (("major", _MAJOR), ("minor", _MINOR)):
            r = float(np.corrcoef(chroma_sum, np.roll(prof, pc))[0, 1])
            scores.append((r, pc, mode))
    scores.sort(reverse=True)
    r, pc, mode = scores[0]
    ranked = [(key_name(p, m), round(rr, 3)) for rr, p, m in scores[:4]]
    return KeyInfo(tonic_pc=pc, mode=mode, confidence=round(r, 3), ranked=ranked)


@dataclass
class Beats:
    bpm: float
    times: np.ndarray              # beat times in seconds

    def to_beats(self, t) -> np.ndarray:
        """Seconds -> beat index (float), interpolating between detected beats and
        extrapolating at the median beat period outside them (handles drift)."""
        t = np.asarray(t, dtype=float)
        bt = self.times
        idx = np.arange(len(bt), dtype=float)
        period = float(np.median(np.diff(bt))) if len(bt) > 1 else 60.0 / self.bpm
        out = np.interp(t, bt, idx)
        out = np.where(t < bt[0], (t - bt[0]) / period, out)
        out = np.where(t > bt[-1], idx[-1] + (t - bt[-1]) / period, out)
        return out


def detect_beats(wav: Path, bpm_hint: Optional[float] = None, sr: int = 22050) -> Beats:
    import librosa
    y, _ = librosa.load(str(wav), sr=sr, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr)
    kw = {"bpm": float(bpm_hint)} if bpm_hint else {}
    tempo, frames = librosa.beat.beat_track(onset_envelope=env, sr=sr, units="frames", **kw)
    times = librosa.frames_to_time(frames, sr=sr)
    if len(times) < 2:                                    # no pulse found: synthesize a grid
        bpm = float(bpm_hint or 120.0)
        times = np.arange(0, len(y) / sr, 60.0 / bpm)
    period = float(np.median(np.diff(times)))
    return Beats(bpm=round(60.0 / period, 1), times=times)


def pulse_source(stems: Dict[str, Path], levels: Dict[str, float], mix: Path) -> Path:
    """Drums stem if it carries real energy, else the full mix."""
    if "drums" in stems and levels.get("drums", -99) > -20:
        return stems["drums"]
    return mix
