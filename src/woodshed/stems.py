"""Source separation with Demucs (htdemucs_6s: drums, bass, other, vocals, guitar,
piano). Uses the Python API and writes stems with soundfile, which sidesteps the
torchaudio/torchcodec save path (same approach as the engine's demucs spike).

Horns, flute, clarinet, strings and synth leads usually land in `other` (or
`vocals` when they are the lead); `piano` is the weakest of the six.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np

MODEL = "htdemucs_6s"
STEMS = ("vocals", "other", "piano", "guitar", "bass", "drums")


def stem_paths(run_dir: Path) -> Dict[str, Path]:
    return {s: run_dir / "stems" / f"{s}.wav" for s in STEMS}


def separate(mix_wav: Path, run_dir: Path, log: Callable[[str], None] = print) -> Dict[str, Path]:
    """mix.wav -> {stem: wav}. Cached: skipped if all stems already exist."""
    paths = stem_paths(run_dir)
    if all(p.exists() for p in paths.values()):
        return paths
    import librosa
    import soundfile as sf
    import torch
    from demucs.apply import apply_model
    from demucs.pretrained import get_model

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Separating stems with {MODEL} on {device.upper()} (first run downloads the model)...")
    model = get_model(MODEL)
    model.eval()
    y = librosa.load(str(mix_wav), sr=model.samplerate, mono=False)[0]
    if y.ndim == 1:
        y = np.stack([y, y])
    wav = torch.from_numpy(y).float()
    ref = wav.mean(0)                                   # demucs' own input normalization
    mean, std = ref.mean(), ref.std() + 1e-8
    with torch.no_grad():
        out = apply_model(model, ((wav - mean) / std)[None], device=device,
                          split=True, overlap=0.25, progress=False)[0]
    out = out * std + mean
    (run_dir / "stems").mkdir(exist_ok=True)
    for i, name in enumerate(model.sources):
        sf.write(paths[name], out[i].cpu().numpy().T, model.samplerate)
    if device == "cuda":
        torch.cuda.empty_cache()
    return paths


def stem_levels(paths: Dict[str, Path], mix_wav: Optional[Path] = None) -> Dict[str, float]:
    """RMS of each stem in dB relative to the loudest stem - tells you which stems
    actually have something in them (a -40 dB 'guitar' stem is just bleed)."""
    import soundfile as sf
    rms = {}
    for name, p in paths.items():
        y, _ = sf.read(p, always_2d=True)
        rms[name] = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    top = max(rms.values())
    return {k: round(float(20 * np.log10(v / top)), 1) for k, v in rms.items()}
