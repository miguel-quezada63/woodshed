"""Runs INSIDE the quarantined .venv-bp (TensorFlow). Reads one or more wav paths,
runs basic-pitch on each, writes {wav_path: [note,...]} as JSON to an OUTPUT FILE
(not stdout - TF/keras pollute stdout). Batching amortizes the slow TF cold start.
Invoked as a subprocess by src/symbolic/extraction.py so TensorFlow never touches
the engine env.

Usage: .venv-bp/Scripts/python.exe tools/bp_runner.py <out.json> <wav> [<wav> ...]
"""
import json
import logging
import os
import sys

logging.disable(logging.WARNING)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")


def main():
    out_path = sys.argv[1]
    wavs = sys.argv[2:]
    from basic_pitch.inference import predict
    try:
        from basic_pitch import ICASSP_2022_MODEL_PATH
        model_path = ICASSP_2022_MODEL_PATH
    except Exception:
        model_path = None

    out = {}
    for wav in wavs:
        note_events = (predict(wav, model_path) if model_path else predict(wav))[2]
        # ev = (start, end, pitch_midi, amplitude, [pitch_bends])
        out[wav] = [{"midi": int(ev[2]), "onset": float(ev[0]), "offset": float(ev[1]),
                     "amp": float(ev[3]) if len(ev) > 3 else 1.0}
                    for ev in note_events]
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
