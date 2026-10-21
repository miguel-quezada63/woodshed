"""link/file -> stems -> key/Camelot/BPM -> per-stem transcription -> sheet music.

Every expensive step is cached in the run dir (mix, stems, raw per-stem notes,
key), so changing grid / downbeat / BPM / flute-fit only re-quantizes and
re-engraves - seconds, not minutes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from . import analyze, fetch, notation, stems, transcribe

DEFAULT_PARTS = ("other", "vocals", "bass")


@dataclass
class Result:
    run_dir: Path
    title: str
    key: analyze.KeyInfo
    beats: analyze.Beats
    levels: Dict[str, float]
    stems: Dict[str, Path]
    note_counts: Dict[str, int] = field(default_factory=dict)
    part_svgs: Dict[str, List[str]] = field(default_factory=dict)      # label -> svg pages
    files: List[Path] = field(default_factory=list)

    def summary(self) -> str:
        k = self.key
        alts = ", ".join(f"{n} ({r})" for n, r in k.ranked[1:3])
        lines = [
            f"{self.title}",
            f"  Key      {k.name}  (confidence {k.confidence}; also: {alts})",
            f"  Camelot  {k.camelot}  (mixes with {', '.join(analyze.camelot_neighbors(k.camelot))})",
            f"  Relative {k.relative}",
            f"  Tempo    {self.beats.bpm} BPM",
            "  Stems    " + "  ".join(f"{s} {v:+.0f}dB" for s, v in self.levels.items()),
            "  Notes    " + "  ".join(f"{s}:{n}" for s, n in self.note_counts.items()),
            f"  Output   {self.run_dir / 'out'}",
        ]
        return "\n".join(lines)


def _print_log(msg: str, frac: Optional[float] = None) -> None:
    print(f"\r[{frac:4.0%}] {msg:<70}" if frac is not None else f"\n{msg}", end="", flush=True)


def process(source: str, start: Optional[float] = None, end: Optional[float] = None,
            parts: Sequence[str] = DEFAULT_PARTS, grid: str = "16th", downbeat: int = 0,
            bpm: Optional[float] = None, flute_fit: bool = True,
            log: Callable[[str, Optional[float]], None] = _print_log) -> Result:
    """`log(message, fraction_done)` - fraction is 0..1 for the whole job."""
    if grid not in transcribe.GRIDS:
        raise ValueError(f"grid must be one of {transcribe.GRIDS}")
    log("Fetching audio...", 0.0)
    run_dir, mix, title = fetch.fetch(source, start, end)
    import soundfile as sf
    secs = sf.info(str(mix)).duration
    length = f"{int(secs // 60)}:{int(secs % 60):02d}"
    span = f"full track ({length})" if start is None and end is None else f"{length} section"
    log(f"Separating stems - {span}...", 0.03)
    stem_wavs = stems.separate(mix, run_dir, log=lambda m: log(m, 0.03))
    levels = stems.stem_levels(stem_wavs)

    key_cache = run_dir / "key.json"
    if key_cache.exists():
        d = json.loads(key_cache.read_text(encoding="utf-8"))
        key = analyze.KeyInfo(d["tonic_pc"], d["mode"], d["confidence"], [tuple(x) for x in d["ranked"]])
    else:
        log("Detecting key...", 0.15)
        pitched = [p for s, p in stem_wavs.items() if s != "drums" and levels[s] > -30]
        key = analyze.detect_key(pitched)
        key_cache.write_text(json.dumps(key.__dict__), encoding="utf-8")

    log("Tracking beats...", 0.18)
    beats = analyze.detect_beats(analyze.pulse_source(stem_wavs, levels, mix), bpm_hint=bpm)

    quantized = {}
    t0, t1 = 0.2, 0.92                                   # transcription's share of the bar
    for i, stem in enumerate(parts):
        lo, hi = t0 + (t1 - t0) * i / len(parts), t0 + (t1 - t0) * (i + 1) / len(parts)
        msg = f"Transcribing {stem} ({i + 1}/{len(parts)}, {span})"
        log(msg, lo)
        raw = transcribe.transcribe_stem(stem_wavs[stem], stem, cache=run_dir / "notes" / f"{stem}.json",
                                         on_progress=lambda f, lo=lo, hi=hi, msg=msg: log(msg, lo + (hi - lo) * f))
        quantized[stem] = transcribe.quantize(raw, beats, downbeat=downbeat, grid=grid)
    song = transcribe.build_song(title, key, beats, quantized, grid)

    log("Engraving sheet music...", t1)
    out = run_dir / "out"
    out.mkdir(exist_ok=True)
    res = Result(run_dir=run_dir, title=title, key=key, beats=beats, levels=levels, stems=stem_wavs,
                 note_counts={s: len(q) for s, q in quantized.items()})
    song_json = out / "song.json"
    song_json.write_text(song.model_dump_json(exclude_none=True, indent=2), encoding="utf-8")
    swing = grid == "swing8"
    full = notation.build_score(song, list(parts), flute_fit=flute_fit, swing=swing)
    res.files += [notation.write_musicxml(full, out / "score.musicxml"),
                  notation.write_midi(full, out / "score.mid"), song_json]
    for stem in parts:
        if not quantized[stem]:
            continue
        score = notation.build_score(song, [stem], flute_fit=flute_fit, swing=swing)
        label = score.parts[0].partName
        xml = notation.write_musicxml(score, out / f"{stem}.musicxml")
        res.files.append(xml)
        res.part_svgs[label] = notation.render_svg(xml)
    log("Done", 1.0)
    return res
