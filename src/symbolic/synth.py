"""Multi-instrument render - the disposable audio layer, a nicer stub than pure
sines. Additive-synthesis instrument presets are mapped per track (melody / chords
/ bass) so a Song can be previewed with varied timbres. The symbolic model stays
the single source of truth; for *real* instruments, export MIDI to a DAW
(io_render.export_midi) - this is only for fast, dependency-free previews.

Each preset: harmonic spectrum, ADSR envelope (sec), and optional vibrato
(depth in semitones, rate in Hz).
"""
from __future__ import annotations

import numpy as np

from .model import Song

INSTRUMENTS = {
    # breathy flutes: near-sine + airy breath noise
    "flute":   {"harm": [(1, 1.0), (2, 0.2), (3, 0.06)], "adsr": (0.05, 0.10, 0.85, 0.18),
                "vib": (0.06, 5.0), "amp": 0.55, "detune": 0, "noise": 0.05, "noise_mode": "breath"},
    "dizi":    {"harm": [(1, 1.0), (2, 0.45), (3, 0.2), (4, 0.08)], "adsr": (0.03, 0.08, 0.85, 0.15),
                "vib": (0.09, 5.5), "amp": 0.5, "detune": 0, "noise": 0.11, "noise_mode": "breath"},
    # plucked: bright attack-noise burst + fast harmonic decay (sustain 0)
    "koto":    {"harm": [(1, 1.0), (2, 0.5), (3, 0.4), (4, 0.25), (5, 0.15)], "adsr": (0.003, 0.45, 0.0, 0.25),
                "vib": (0, 0), "amp": 0.5, "detune": 0, "noise": 0.25, "noise_mode": "attack"},
    # lush detuned-unison saw (1/k harmonics ~ sawtooth), 3 voices
    "strings": {"harm": [(k, 1.0 / k) for k in range(1, 11)], "adsr": (0.28, 0.2, 0.85, 0.5),
                "vib": (0.04, 5.5), "amp": 0.20, "detune": 11, "noise": 0.0},
    # wide warm pad: heavy detune, slow attack/release, fewer highs
    "pad":     {"harm": [(1, 1.0), (2, 0.5), (3, 0.28), (4, 0.16), (5, 0.09)], "adsr": (0.5, 0.3, 0.7, 0.7),
                "vib": (0.03, 3.5), "amp": 0.24, "detune": 16, "noise": 0.0},
    # triangle-ish (odd harmonics) + body
    "bass":    {"harm": [(1, 1.0), (3, 0.25), (5, 0.1)], "adsr": (0.02, 0.12, 0.85, 0.2),
                "vib": (0, 0), "amp": 0.62, "detune": 0, "noise": 0.0},
    # metallic inharmonic partials
    "bell":    {"harm": [(1, 1.0), (2.76, 0.55), (5.40, 0.28), (8.93, 0.12)], "adsr": (0.003, 1.4, 0.0, 0.8),
                "vib": (0, 0), "amp": 0.4, "detune": 0, "noise": 0.06, "noise_mode": "attack"},
    # e-piano: sine + 4th-harmonic bell + tine click
    "rhodes":  {"harm": [(1, 1.0), (2, 0.25), (4, 0.5), (6, 0.08)], "adsr": (0.004, 0.5, 0.3, 0.4),
                "vib": (0, 0), "amp": 0.5, "detune": 0, "noise": 0.08, "noise_mode": "attack"},
    "sine":    {"harm": [(1, 1.0)], "adsr": (0.01, 0.05, 0.9, 0.05), "vib": (0, 0),
                "amp": 0.5, "detune": 0, "noise": 0.0},
    # punchy sub for bass-register leads: each note decays to silence (sustain 0)
    # so contiguous same-pitch re-articulations stay distinct instead of blurring
    "sub808":  {"harm": [(1, 1.0), (2, 0.3), (3, 0.08)], "adsr": (0.002, 0.22, 0.0, 0.06),
                "vib": (0, 0), "amp": 0.72, "detune": 0, "noise": 0.04, "noise_mode": "attack"},
}

# named instrument ensembles (melody / chords / bass / chord-style) for previews
ENSEMBLES = {
    "eastern":   dict(melody="dizi", chords="koto", bass="bass", chord_style="arp"),
    "cinematic": dict(melody="flute", chords="strings", bass="bass", chord_style="pad"),
    "lofi":      dict(melody="rhodes", chords="pad", bass="bass", chord_style="block"),
    "chamber":   dict(melody="flute", chords="strings", bass="bass", chord_style="arp"),
    "default":   dict(melody="flute", chords="pad", bass="bass", chord_style="arp"),
}


def _adsr(n, sr, dur, a, d, s, r):
    env = np.zeros(n, dtype=np.float32)
    ai = min(n, int(a * sr))                       # clip every stage to what fits
    di = min(n - ai, int(d * sr))
    sus_end = min(n, max(ai + di, int(dur * sr)))
    ri = int(r * sr)
    if ai:
        env[:ai] = np.linspace(0, 1, ai)
    if di:
        env[ai:ai + di] = np.linspace(1, s, di)
    if sus_end > ai + di:
        env[ai + di:sus_end] = s
    rel_end = min(n, sus_end + ri)
    if rel_end > sus_end:
        env[sus_end:rel_end] = np.linspace(s, 0, rel_end - sus_end)
    return env


def _note(freq, dur, sr, preset, a4=440.0):
    a, d, s, r = preset["adsr"]
    n = max(1, int((dur + r) * sr))
    t = np.arange(n) / sr
    env = _adsr(n, sr, dur, a, d, s, r)
    vd, vr = preset["vib"]

    def osc(f):
        if vd > 0:
            fmod = f * 2 ** ((vd * np.sin(2 * np.pi * vr * t)) / 12.0)
            ph = 2 * np.pi * np.cumsum(fmod) / sr
        else:
            ph = 2 * np.pi * f * t
        return sum(amp * np.sin(ratio * ph) for ratio, amp in preset["harm"])

    det = preset.get("detune", 0)
    if det > 0:                                            # detuned unison (3 voices)
        wave = (osc(freq * 2 ** (-det / 1200)) + osc(freq) + osc(freq * 2 ** (det / 1200))) / 3.0
    else:
        wave = osc(freq)
    out = wave * env

    nz = preset.get("noise", 0.0)
    if nz > 0:                                             # breath / pluck-attack noise
        noise = np.random.RandomState(int(freq) % 10000).randn(n).astype(np.float32)
        nenv = np.exp(-t / 0.02) if preset.get("noise_mode") == "attack" else env
        out = out + nz * noise * nenv
    return (preset["amp"] * out).astype(np.float32)


def _hz(midi, a4=440.0):
    return a4 * 2 ** ((midi - 69) / 12)


def _mix_write(events, path, sr, reverb):
    """events: list of (onset_sec, dur_sec, midi, instrument). Render, add reverb,
    soft-clip master, write."""
    total = max((o + d for o, d, _, _ in events), default=1.0) + 0.6
    buf = np.zeros(int(total * sr) + sr, dtype=np.float32)
    for on, du, midi, inst in events:
        w = _note(_hz(midi), max(du, 0.05), sr, INSTRUMENTS.get(inst, INSTRUMENTS["sine"]))
        i = int(on * sr)
        end = min(len(buf), i + len(w))
        buf[i:end] += w[: end - i]
    if reverb > 0:
        from scipy.signal import fftconvolve
        ir_len = int(0.45 * sr)
        t = np.arange(ir_len) / sr
        ir = (np.random.RandomState(0).randn(ir_len) * np.exp(-t / 0.13)).astype(np.float32)
        wet = fftconvolve(buf, ir)[: len(buf)]
        wet *= (np.max(np.abs(buf)) or 1.0) / (np.max(np.abs(wet)) or 1.0)
        buf = (1 - reverb) * buf + reverb * wet
    peak = float(np.max(np.abs(buf))) or 1.0
    buf = np.tanh(1.2 * buf / peak) * 0.9
    import soundfile as sf
    sf.write(path, buf.astype(np.float32), sr)
    return path


def render_multitrack(song: Song, path: str, voice_instruments=None, sr: int = 22050,
                      reverb: float = 0.22) -> str:
    """Render a multi-track Song (e.g. from the polyphonic path): each melody-kind
    track is voiced by its voiceId via `voice_instruments`."""
    vi = voice_instruments or {"lead": "flute", "inner": "rhodes", "bass": "sub808"}
    ppq, bpm = song.ppq, song.tempo_bpm()
    beat = 60.0 / bpm
    events = []
    for sec in song.sections:
        for trk in sec.tracks:
            if trk.kind == "melody" and trk.notes:
                inst = vi.get(trk.voiceId or "", vi.get("lead", "flute"))
                for n in trk.notes:
                    if n.measured and n.measured.onsetSec is not None:
                        on, du = n.measured.onsetSec, (n.measured.durationSec or 0.2)
                    else:
                        on = (sec.startTick + n.startTick) / ppq * beat
                        du = n.durationTick / ppq * beat
                    events.append((on, du, n.pitch.midi, inst))
    return _mix_write(events, path, sr, reverb)


def render_song(song: Song, path: str, sr: int = 22050, melody: str = "flute",
                chords: str = "strings", bass: str = "bass", chord_style: str = "arp",
                reverb: float = 0.25, use_measured_melody: bool = True) -> str:
    """Render melody + chords (+ derived bass) with chosen instruments.
    chord_style: 'arp' (8th-note arpeggio), 'pad'/'block' (held chord)."""
    ppq, bpm = song.ppq, song.tempo_bpm()
    beat = 60.0 / bpm
    events = []  # (onset_sec, dur_sec, midi, instrument)

    def t2s(tick):
        return (tick / ppq) * beat

    # bass-register lead (e.g. a contrabassoon trap riff): render the melody as a punchy
    # sub (contiguous re-articulations stay distinct instead of blurring) and drop the bass.
    mel_all = [n.pitch.midi for _, _, n in song.iter_melody_notes()]
    if mel_all and float(np.median(mel_all)) < 48:
        melody, bass = "sub808", None

    for sec, trk, nt in song.iter_melody_notes():
        if use_measured_melody and nt.measured and nt.measured.onsetSec is not None:
            on, du = nt.measured.onsetSec, (nt.measured.durationSec or 0.2)
        else:
            on, du = t2s(sec.startTick + nt.startTick), t2s(nt.durationTick)
        events.append((on, du, nt.pitch.midi, melody))

    for sec in song.sections:
        for trk in sec.tracks:
            if trk.kind != "harmony" or not trk.chords:
                continue
            for ch in trk.chords:
                on, du = t2s(sec.startTick + ch.startTick), t2s(ch.durationTick)
                pcs = ch.chordPitchClasses or []
                tones = sorted(60 + pc for pc in pcs)        # chords voiced in octave 4
                if chord_style in ("pad", "block"):
                    for m in tones:
                        events.append((on, du, m, chords))
                else:                                        # arpeggio (up-down)
                    seq = tones + tones[-2:0:-1] or tones
                    step = beat / 2
                    k, tpos = 0, on
                    while tpos < on + du - 1e-3 and seq:
                        events.append((tpos, min(step * 1.1, on + du - tpos), seq[k % len(seq)], chords))
                        tpos += step
                        k += 1
                if bass:
                    root = ch.root.pitchClass if ch.root else (pcs[0] if pcs else 0)
                    events.append((on, du, 36 + root, bass))   # bass in octave 2

    return _mix_write(events, path, sr, reverb)
