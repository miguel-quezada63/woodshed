"""Spike D - constrained variation via LLM-as-symbolic-generator.

The melody is fed to the model as structured JSON; the model must return a
variation as structured JSON (schema-enforced via Ollama `format`). We then
VALIDATE the result against the musical constraints (in-key, in-range, motif
preserved, length preserved) and deterministically REPAIR violations. The thesis
under test: an LLM can produce coherent symbolic edits if (and only if) it is
wrapped in a validate-and-repair loop - the loop is the product, the model is a
component.

Backends: Ollama (local, schema-enforced) and Cerebras (OpenAI-compatible) if
CEREBRAS_API_KEY is set. Keys/hosts come from env vars only.
"""
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Tuple

import httpx

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "midi": {"type": "integer"},
                    "durationTicks": {"type": "integer"},
                },
                "required": ["midi", "durationTicks"],
            },
        }
    },
    "required": ["notes"],
}


# ----------------------------------------------------------------- backends
def _ollama_url() -> str:
    host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434").replace("0.0.0.0", "127.0.0.1")
    if not host.startswith("http"):
        host = "http://" + host
    return host.rstrip("/")


def generate_ollama(model: str, messages: List[Dict], temperature: float,
                    seed: int, timeout: float = 120.0) -> Tuple[str, float]:
    body = {
        "model": model, "messages": messages, "stream": False,
        "format": OUTPUT_SCHEMA,
        "options": {"temperature": temperature, "seed": seed},
    }
    t0 = time.perf_counter()
    r = httpx.post(f"{_ollama_url()}/api/chat", json=body, timeout=timeout)
    r.raise_for_status()
    dt = time.perf_counter() - t0
    return r.json()["message"]["content"], dt


def generate_cerebras(model: str, messages: List[Dict], temperature: float,
                      seed: int, timeout: float = 120.0) -> Tuple[str, float]:
    key = os.environ["CEREBRAS_API_KEY"]
    body = {"model": model, "messages": messages, "temperature": temperature,
            "seed": seed, "response_format": {"type": "json_object"}}
    t0 = time.perf_counter()
    r = httpx.post("https://api.cerebras.ai/v1/chat/completions", json=body,
                   headers={"Authorization": f"Bearer {key}"}, timeout=timeout)
    r.raise_for_status()
    dt = time.perf_counter() - t0
    return r.json()["choices"][0]["message"]["content"], dt


def ollama_alive() -> bool:
    try:
        httpx.get(f"{_ollama_url()}/api/tags", timeout=4).raise_for_status()
        return True
    except Exception:
        return False


# ----------------------------------------------------------------- prompt
def build_messages(phrase: List[Dict], ctx: Dict) -> List[Dict]:
    scale = ctx["scale_pcs"]
    motif = phrase[: ctx["motif_len"]]
    system = (
        "You are a music composition assistant that edits melodies represented as "
        "structured JSON. You MUST return only JSON matching the requested schema. "
        "A melody is a list of notes; each note has an integer MIDI pitch and an "
        "integer durationTicks. Respect every constraint exactly."
    )
    user = {
        "task": "Compose a melodic VARIATION of the given phrase.",
        "constraints": {
            "stay_in_scale_pitch_classes": scale,
            "pitch_range_midi": [ctx["minMidi"], ctx["maxMidi"]],
            "preserve_exactly_the_first_N_notes": ctx["motif_len"],
            "total_duration_ticks_must_equal": ctx["total_ticks"],
            "keep_similar_contour_and_rhythm": True,
            "ppq": ctx["ppq"],
        },
        "key": ctx["key"],
        "opening_motif_to_keep": motif,
        "original_phrase": phrase,
        "instructions": (
            "Vary the notes AFTER the opening motif: change some pitches and/or "
            "rhythm while keeping every pitch within the given scale and range, and "
            "keeping the total duration identical. Avoid large leaps (> a fifth). "
            "Return JSON: {\"notes\": [{\"midi\": int, \"durationTicks\": int}, ...]}."
        ),
    }
    return [{"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user)}]


# ----------------------------------------------------------------- validate / repair
def _nearest_in_scale(midi: int, scale: List[int]) -> int:
    for d in range(0, 12):
        for cand in (midi - d, midi + d):
            if cand % 12 in scale:
                return cand
    return midi


def validate(notes: List[Dict], orig: List[Dict], ctx: Dict) -> Dict:
    scale = set(ctx["scale_pcs"])
    in_key = all(n["midi"] % 12 in scale for n in notes) if notes else False
    in_range = all(ctx["minMidi"] <= n["midi"] <= ctx["maxMidi"] for n in notes) if notes else False
    k = ctx["motif_len"]
    motif_ok = (len(notes) >= k and all(
        notes[i]["midi"] == orig[i]["midi"] and notes[i]["durationTicks"] == orig[i]["durationTicks"]
        for i in range(k)))
    total = sum(n["durationTicks"] for n in notes)
    len_ok = total == ctx["total_ticks"]
    nonempty = len(notes) > 0
    return {"in_key": in_key, "in_range": in_range, "motif_ok": motif_ok,
            "len_ok": len_ok, "nonempty": nonempty,
            "valid": all((in_key, in_range, motif_ok, len_ok, nonempty))}


def repair(notes: List[Dict], orig: List[Dict], ctx: Dict) -> List[Dict]:
    scale = ctx["scale_pcs"]
    fixed = []
    for n in notes:
        m = int(n.get("midi", ctx["minMidi"]))
        d = max(1, int(n.get("durationTicks", ctx["ppq"])))
        m = _nearest_in_scale(m, scale)
        m = min(ctx["maxMidi"], max(ctx["minMidi"], m))
        # snapping to range may leave it out of scale; re-snap within range
        if m % 12 not in scale:
            m = _nearest_in_scale(m, scale)
            m = min(ctx["maxMidi"], max(ctx["minMidi"], m))
        fixed.append({"midi": m, "durationTicks": d})
    # force the opening motif
    k = ctx["motif_len"]
    fixed = [dict(orig[i]) for i in range(k)] + (fixed[k:] if len(fixed) > k else [])
    if not fixed:
        fixed = [dict(o) for o in orig]
    # force total duration onto the last (non-motif) note
    target = ctx["total_ticks"]
    diff = target - sum(n["durationTicks"] for n in fixed)
    if diff != 0:
        if fixed[-1]["durationTicks"] + diff >= 1 and len(fixed) > k:
            fixed[-1]["durationTicks"] += diff
        else:
            fixed = [dict(o) for o in orig]  # last resort: identity
    return fixed


def coherence(orig: List[Dict], var: List[Dict], scale: List[int]) -> Dict:
    """Lightweight musical-coherence proxies (post-repair)."""
    sset = set(scale)
    in_scale = sum(1 for n in var if n["midi"] % 12 in sset) / max(1, len(var))
    ivals = [abs(var[i + 1]["midi"] - var[i]["midi"]) for i in range(len(var) - 1)]
    smooth = (sum(1 for x in ivals if x <= 7) / len(ivals)) if ivals else 1.0  # <= P5
    changed = sum(1 for i in range(min(len(orig), len(var)))
                  if orig[i]["midi"] != var[i]["midi"]) / max(1, len(var))
    return {"in_scale_rate": in_scale, "stepwise_smoothness": smooth, "novelty": changed}
