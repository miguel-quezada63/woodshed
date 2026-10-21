"""Get the audio: a link (YouTube or anything yt-dlp supports) or a local file ->
a stereo 44.1 kHz wav in its own run directory, optionally trimmed to a section.

Each (source, start, end) gets its own run dir so stems/transcriptions are cached
and re-notating with different settings is instant.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Optional, Tuple

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
DOWNLOADS = RUNS / "_downloads"


def parse_time(s) -> Optional[float]:
    """'1:23' / '83' / '1:02:03' / '' -> seconds (None if blank)."""
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    sec = 0.0
    for part in s.split(":"):
        sec = sec * 60 + float(part)
    return sec


def clean_title(text: str) -> str:
    """'Artist - Song (Official MV) [abc123]' -> 'Artist - Song'."""
    return re.sub(r"\s+", " ", re.sub(r"\[[^\]]*\]|\([^)]*\)", "", text)).strip() or text


def slug(text: str, n: int = 48) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "-", clean_title(text)).strip("-").lower()
    return text[:n].strip("-") or "song"


def is_url(source: str) -> bool:
    return bool(re.match(r"^https?://", source.strip(), re.I))


def download(url: str) -> Tuple[Path, str]:
    """yt-dlp best audio -> (file, title). Re-uses an earlier download of the same id."""
    import yt_dlp
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    opts = {"format": "bestaudio/best", "outtmpl": str(DOWNLOADS / "%(id)s.%(ext)s"),
            "quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        cached = sorted(DOWNLOADS.glob(f"{info['id']}.*"))
        cached = [p for p in cached if p.suffix != ".json"]
        if cached:
            return cached[0], info.get("title") or info["id"]
        info = ydl.extract_info(url, download=True)
        return Path(ydl.prepare_filename(info)), info.get("title") or info["id"]


def fetch(source: str, start: Optional[float] = None, end: Optional[float] = None) -> Tuple[Path, Path, str]:
    """-> (run_dir, mix_wav, title). mix.wav is stereo 44.1 kHz (what Demucs wants)."""
    source = source.strip().strip('"')
    if is_url(source):
        src, title = download(source)
    else:
        src = Path(source)
        if not src.exists():
            raise FileNotFoundError(f"No such file: {src}")
        title = src.stem

    span = ""
    if start is not None or end is not None:
        span = f"_{int(start or 0)}-{int(end) if end is not None else 'end'}"
    norm = lambda t: None if t is None else round(float(t), 2)       # 30 == 30.0
    digest = hashlib.sha1(f"{source}|{norm(start)}|{norm(end)}".encode()).hexdigest()[:6]
    run_dir = RUNS / f"{slug(title)}{span}_{digest}"
    run_dir.mkdir(parents=True, exist_ok=True)
    mix = run_dir / "mix.wav"
    if not mix.exists():
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        if start:
            cmd += ["-ss", str(start)]
        cmd += ["-i", str(src)]
        if end is not None:
            cmd += ["-t", str(end - (start or 0))]
        cmd += ["-ac", "2", "-ar", "44100", str(mix)]
        subprocess.run(cmd, check=True, capture_output=True)
        (run_dir / "source.json").write_text(json.dumps(
            {"source": source, "title": title, "start": start, "end": end}, indent=2), encoding="utf-8")
    return run_dir, mix, clean_title(title)
