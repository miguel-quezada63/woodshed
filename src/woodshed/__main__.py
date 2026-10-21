"""CLI: woodshed <link-or-file> [--start 0:30 --end 1:30 --parts other,bass --grid swing8]"""
from __future__ import annotations

import argparse

from . import fetch, pipeline, transcribe


def main(argv=None):
    ap = argparse.ArgumentParser(prog="woodshed", description=__doc__)
    ap.add_argument("source", help="YouTube (or any yt-dlp) link, or a local audio file")
    ap.add_argument("--start", help="section start, e.g. 0:45")
    ap.add_argument("--end", help="section end, e.g. 1:30")
    ap.add_argument("--parts", default=",".join(pipeline.DEFAULT_PARTS),
                    help="stems to transcribe: vocals,other,piano,guitar,bass")
    ap.add_argument("--grid", default="16th", choices=transcribe.GRIDS)
    ap.add_argument("--downbeat", type=int, default=0, help="which detected beat (0-3) is beat 1")
    ap.add_argument("--bpm", type=float, help="tempo hint if detection picks half/double time")
    ap.add_argument("--concert", action="store_true", help="keep original octaves (no flute fit)")
    a = ap.parse_args(argv)
    res = pipeline.process(a.source, fetch.parse_time(a.start), fetch.parse_time(a.end),
                           parts=[p.strip() for p in a.parts.split(",") if p.strip()],
                           grid=a.grid, downbeat=a.downbeat, bpm=a.bpm, flute_fit=not a.concert)
    print()
    print(res.summary())


if __name__ == "__main__":
    main()
