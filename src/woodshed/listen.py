"""Hear the transcription, to check it by ear.

For each part, two wavs built from the QUANTIZED notes (exactly what the sheet
shows), mapped back onto the song's beat grid so they line up with the record:
  <part>_transcription.wav  the transcription alone (synth flute / bass)
  <part>_check.wav          the original stem with the transcription on top -
                            right notes blend in, wrong ones clash
Rendered at concert pitch (no flute-range octave shift) so it matches the stem.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from symbolic import synth

from .analyze import Beats
from .transcribe import score_to_seconds

SR = 22050
VOICE = {"bass": "bass"}            # synth preset per stem; everything else plays as flute


def _render(events: List[Tuple[float, float, int]], n: int, inst: str) -> np.ndarray:
    buf = np.zeros(n, dtype=np.float32)
    preset = synth.INSTRUMENTS[inst]
    lead = 0.75 * preset["adsr"][0]          # start early by most of the attack so the
    for on, du, midi in events:              # note is *heard* on the beat, not late
        w = synth._note(synth._hz(midi), max(du, 0.05), SR, preset)
        i = max(0, int((on - lead) * SR))
        if i >= n:
            continue
        end = min(n, i + len(w))
        buf[i:end] += w[: end - i]
    return buf


def _norm(x: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(x))) or 1.0
    return (peak * x / m).astype(np.float32)


def render_checks(stem: str, stem_wav: Path, quantized, beats: Beats, downbeat: int,
                  shift: int, out_dir: Path) -> Dict[str, Path]:
    """-> {"transcription": wav, "check": wav} for one part."""
    import librosa
    import soundfile as sf
    orig, _ = librosa.load(str(stem_wav), sr=SR, mono=True)
    events = []
    for start, dur, n in quantized:
        on, off = score_to_seconds([start, start + dur], beats, downbeat, shift)
        events.append((float(on), float(off - on), int(n["midi"])))
    synth_buf = _norm(_render(events, len(orig), VOICE.get(stem, "flute")))
    paths = {"transcription": out_dir / f"{stem}_transcription.wav",
             "check": out_dir / f"{stem}_check.wav"}
    sf.write(paths["transcription"], synth_buf, SR)
    sf.write(paths["check"], _norm(0.8 * _norm(orig) + 0.6 * synth_buf), SR)
    return paths
