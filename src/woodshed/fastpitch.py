"""Chunked, multi-process pYIN: a drop-in for librosa.pyin that uses every CPU
core and reports real progress.

pYIN has no GPU implementation and runs on one core, so a 9-minute stem takes
minutes. Here the signal is cut into chunks with a margin on each side, the
chunks run in a process pool, and each chunk keeps only its own frames (the
margins absorb pYIN's edge and Viterbi-boundary effects). The stitched contour
matches a single full pass apart from rare decisions right at a chunk boundary.
"""
from __future__ import annotations

import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, Optional

import numpy as np

# physical cores minus one, so the UI / Demucs stay responsive
WORKERS = max(1, min(16, (os.cpu_count() or 2) // 2 - 1))
MIN_CHUNK_SEC, MAX_CHUNK_SEC, MARGIN_SEC = 8.0, 30.0, 1.0

_pool: Optional[ProcessPoolExecutor] = None


def pool() -> ProcessPoolExecutor:
    """One long-lived pool per process (workers pay the librosa import once)."""
    global _pool
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=WORKERS)
    return _pool


def _job(y, sr, fmin, fmax, hop, frame_length, kw):
    import librosa
    return librosa.pyin(y, sr=sr, fmin=fmin, fmax=fmax, hop_length=hop,
                        frame_length=frame_length, **kw)


class ChunkedPyin:
    """Callable with librosa.pyin's signature. `on_progress(pass_no, done, total)`
    fires as chunks finish (pass_no counts calls: the engine's auto-register makes
    a coarse pass before the real one)."""

    def __init__(self, on_progress: Optional[Callable[[int, int, int], None]] = None):
        self.on_progress = on_progress
        self.passes = 0

    def __call__(self, y, sr=22050, fmin=65.0, fmax=2093.0, hop_length=512,
                 frame_length=2048, **kw):
        self.passes += 1
        hop = hop_length
        n_frames = 1 + len(y) // hop                     # librosa's centered frame count
        chunk_sec = min(MAX_CHUNK_SEC, max(MIN_CHUNK_SEC, len(y) / sr / WORKERS))
        chunk = max(hop, int(chunk_sec * sr) // hop * hop)
        margin = math.ceil(max(MARGIN_SEC * sr, frame_length) / hop) * hop

        jobs = {}
        for s in range(0, len(y), chunk):
            a, b = max(0, s - margin), min(len(y), s + chunk + margin)
            fut = pool().submit(_job, y[a:b], sr, fmin, fmax, hop, frame_length, kw)
            jobs[fut] = (s, a)

        f0 = np.full(n_frames, np.nan)
        voiced = np.zeros(n_frames, dtype=bool)
        prob = np.zeros(n_frames)
        for done, fut in enumerate(as_completed(jobs), 1):
            s, a = jobs[fut]
            cf, cv, cp = fut.result()
            g0, l0 = s // hop, (s - a) // hop             # global / local first frame
            own = n_frames - g0 if s + chunk >= len(y) else chunk // hop   # last chunk keeps the tail
            n = min(own, len(cf) - l0)
            f0[g0:g0 + n], voiced[g0:g0 + n], prob[g0:g0 + n] = cf[l0:l0 + n], cv[l0:l0 + n], cp[l0:l0 + n]
            if self.on_progress:
                self.on_progress(self.passes, done, len(jobs))
        return f0, voiced, prob
