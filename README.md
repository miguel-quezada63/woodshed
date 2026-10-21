# woodshed

A practice tool for horn and flute players. Paste a song link (or drop in an audio
file) and get:

- **Stems** - the song split into vocals / lead-horns ("other") / piano / guitar / bass / drums (Demucs)
- **Key, Camelot code and tempo** - from the audio, with runner-up keys and confidence
- **Readable sheet music for each part** - quantized to the song's real beat grid,
  swing-aware, moved into flute range by octaves, engraved in the browser (Verovio)
  and exported as MusicXML (opens in MuseScore) + MIDI

> To "woodshed" is jazz slang for going off alone to practice a tune until you own it.

## Quick start

Requires [uv](https://docs.astral.sh/uv/), Python 3.11 and ffmpeg on `PATH`. An NVIDIA
GPU is optional but makes stem separation ~10x faster (CUDA torch is pulled automatically).

```bash
uv sync
uv run python app.py                     # UI at http://127.0.0.1:7860
```

Or from the command line:

```bash
uv run woodshed "https://www.youtube.com/watch?v=..." --start 0:30 --end 1:30 --grid swing8
uv run woodshed song.mp3 --parts other,bass --concert     # keep original octaves
```

Output lands in `runs/<song>/out/`: one MusicXML per part, a full score, MIDI, and the
`song.json` the sheets were built from.

### Settings that matter

| Setting | When to touch it |
|---|---|
| **Start / End** | Transcribe one section (the head, a solo) at a time - faster and easier to check by ear |
| **Rhythm grid** | `swing8` for jazz / electro-swing (notated as straight 8ths + "Swing"), `16th` for straight funk/pop, `8th` for a simpler chart |
| **Shift barline** | If bar 1 starts on the wrong beat, shift by 1-3 beats |
| **Tempo hint** | If the tempo came out half or double what you feel |
| **Flute range** | On: every part moved by whole octaves to sit on the treble staff. Off: concert octaves (bass in bass clef) |

## How it works

```
link/file --yt-dlp/ffmpeg--> mix.wav --Demucs htdemucs_6s--> 6 stems
   stems (no drums) --CQT chroma + Krumhansl-Kessler--> key, Camelot
   drum stem --librosa beat tracker--> beat times, BPM
   each chosen stem --pYIN (auto-register, vibrato/onset-aware)--> notes (seconds)
       --loudness gate (drop bleed)--> --snap to detected beat grid--> Song (ticks)
   Song --music21--> MusicXML / MIDI --Verovio--> SVG sheet music
```

Every expensive step is cached in `runs/`, so changing grid / barline / tempo / range
re-engraves in seconds.

## Honest limits

- **One line per stem.** pYIN follows a single melodic line, which suits vocals, horn/flute
  leads and bass. Piano and guitar give only their top line.
- **Busy mixes blur stems.** A lead buried in a dense production (sampled horns, heavy
  sidechain) separates less cleanly than a small acoustic combo. Check by ear with the stem players.
- **4/4 only** for now, and bar 1 is a guess (use *Shift barline*).
- Transcription is a starting point to read along with the record, not an engraved edition.

## Roadmap

1. ~~Link -> stems -> key/Camelot/BPM -> per-part sheet music~~
2. Chords (detected, or pasted from a chart for jazz standards) -> improv sheet: chord
   scales, arpeggios, guide-tone lines, ii-V-I licks in the song's key
3. Play into the mic -> sheet music of what you played
4. Galaxy Watch (Wear OS) haptic cues: count-ins for entries, off-beat and
   call-and-response cues derived from the beat grid and the gaps in the lead stem

## Origins

The `symbolic` engine (data model, pYIN extraction, theory, MusicXML/MIDI I/O) is forked
from my R&D repo *ai-music*. Its design notes and spike results are in
[`docs/engine/`](docs/engine/). Engine checks: `uv run python spikes/b_roundtrip.py`,
`uv run python spikes/c_transforms.py`.

## Copyright

This is a personal practice tool. Downloaded audio, stems and transcriptions stay in
`runs/`, which is git-ignored. Don't redistribute what it produces from music you don't own.
