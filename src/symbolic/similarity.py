"""Spike E - public-domain closest-match against the Essen Folksong Collection,
which ships inside music21's core corpus (clearly public domain, no scraping, no
copyrighted catalogs). Similarity uses transposition-invariant interval sequences
via two complementary measures: interval n-gram overlap (fast) and DTW on the
interval contour (alignment-aware).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np


@dataclass
class Tune:
    id: str
    name: str
    midis: List[int]
    intervals: List[int] = field(default_factory=list)
    ngrams: set = field(default_factory=set)


def _intervals(midis: List[int]) -> List[int]:
    return [midis[i + 1] - midis[i] for i in range(len(midis) - 1)]


def ngram_set(intervals: List[int], n: int = 3) -> set:
    return {tuple(intervals[i:i + n]) for i in range(len(intervals) - n + 1)}


def load_essen(limit: int = 150, min_notes: int = 8) -> List[Tune]:
    """Load up to `limit` monophonic Essen tunes from the music21 core corpus."""
    from music21 import corpus, converter

    try:
        paths = corpus.getCorePaths()
    except AttributeError:
        paths = corpus.corpora.CoreCorpus().getPaths()
    essen = [p for p in paths if "essen" in str(p).lower()]

    tunes: List[Tune] = []
    for path in essen:
        if len(tunes) >= limit:
            break
        try:
            parsed = converter.parse(path)
        except Exception:
            continue
        scores = list(getattr(parsed, "scores", [])) or [parsed]
        for sc in scores:
            if len(tunes) >= limit:
                break
            midis = [n.pitch.midi for n in sc.recurse().notes if getattr(n, "isNote", False)]
            if len(midis) < min_notes:
                continue
            md = sc.metadata
            name = (md.title if md and md.title else None) or f"essen_{len(tunes)}"
            iv = _intervals(midis)
            tunes.append(Tune(id=f"essen_{len(tunes)}", name=name, midis=midis,
                              intervals=iv, ngrams=ngram_set(iv)))
    return tunes


def ngram_similarity(q: set, t: set) -> float:
    if not q or not t:
        return 0.0
    return len(q & t) / len(q | t)        # Jaccard


def dtw_distance(a: List[int], b: List[int]) -> float:
    """Normalized DTW cost between two interval contours (lower = closer)."""
    import librosa
    if not a or not b:
        return float("inf")
    X = np.asarray(a, dtype=float)[np.newaxis, :]
    Y = np.asarray(b, dtype=float)[np.newaxis, :]
    D, wp = librosa.sequence.dtw(X=X, Y=Y, metric="euclidean", subseq=False)
    return float(D[-1, -1] / len(wp))


def closest(query_midis: List[int], corpus: List[Tune], top_k: int = 3,
            ngram_n: int = 3) -> List[Tuple[Tune, float, float]]:
    """Return top_k (tune, ngram_similarity, dtw_distance), ranked by a blended
    score (n-gram overlap, DTW as tie-break)."""
    qiv = _intervals(query_midis)
    qng = ngram_set(qiv, ngram_n)
    scored = []
    for t in corpus:
        ng = ngram_similarity(qng, t.ngrams)
        dt = dtw_distance(qiv, t.intervals)
        scored.append((t, ng, dt))
    # rank: higher n-gram first, then lower DTW
    scored.sort(key=lambda x: (-x[1], x[2]))
    return scored[:top_k]
