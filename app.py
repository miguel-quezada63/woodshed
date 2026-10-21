"""woodshed UI: paste a song link (or upload audio) -> stems, key/Camelot/BPM, and
readable flute sheet music for each part.

Run:  uv run python app.py    then open http://127.0.0.1:7860
"""
from __future__ import annotations

import gradio as gr

from woodshed import analyze, fetch, pipeline, stems, transcribe

PART_CHOICES = [("Lead / horns / flute (other)", "other"), ("Vocals", "vocals"),
                ("Bass", "bass"), ("Piano (top line)", "piano"), ("Guitar (top line)", "guitar")]
GRID_CHOICES = [("Swing 8ths (jazz, electro-swing)", "swing8"), ("Straight 16ths", "16th"),
                ("Straight 8ths (simpler)", "8th")]

SHEET_CSS = """<style>
.ws-sheet{background:#fff;border-radius:8px;padding:8px 4px;margin:4px 0 18px}
.ws-sheet svg{width:100%;height:auto;display:block}
</style>"""


def summary_md(r: pipeline.Result) -> str:
    k = r.key
    alts = ", ".join(f"{n} ({c})" for n, c in k.ranked[1:3])
    levels = " · ".join(f"{s} {v:+.0f} dB" for s, v in r.levels.items())
    counts = " · ".join(f"{s}: {n}" for s, n in r.note_counts.items())
    return (
        f"## {r.title}\n\n"
        f"| Key | Camelot | Relative | Tempo |\n|---|---|---|---|\n"
        f"| **{k.name}** | **{k.camelot}** (mixes with {', '.join(analyze.camelot_neighbors(k.camelot))}) "
        f"| {k.relative} | **{r.beats.bpm:.0f} BPM** |\n\n"
        f"Key confidence {k.confidence} - runners-up: {alts}\n\n"
        f"**Stem levels:** {levels}  \n_(a stem far below 0 dB is mostly bleed - skip it)_\n\n"
        f"**Notes transcribed:** {counts}"
    )


def sheets_html(r: pipeline.Result) -> str:
    if not r.part_svgs:
        return "<p>No notes found in the selected stems.</p>"
    out = [SHEET_CSS]
    for label, pages in r.part_svgs.items():
        out.append(f"<h3>{label}</h3><div class='ws-sheet'>{''.join(pages)}</div>")
    return "".join(out)


def run(link, upload, start, end, parts, grid, downbeat, bpm, flute_fit, progress=gr.Progress()):
    source = upload or (link or "").strip()
    empty = [None] * len(stems.STEMS)
    if not source:
        return ["Paste a link or upload a file first.", "", None, *empty]
    if not parts:
        return ["Pick at least one part to transcribe.", "", None, *empty]
    try:
        r = pipeline.process(source, fetch.parse_time(start), fetch.parse_time(end),
                             parts=parts, grid=grid, downbeat=int(downbeat),
                             bpm=float(bpm) if bpm else None, flute_fit=flute_fit,
                             log=lambda m, f=None: progress(f, desc=m))
    except Exception as e:                                   # surface, don't crash the UI
        return [f"**Error:** {type(e).__name__}: {e}", "", None, *empty]
    return [summary_md(r), sheets_html(r), [str(f) for f in r.files],
            *[str(r.stems[s]) for s in stems.STEMS]]


def build() -> gr.Blocks:
    with gr.Blocks(title="woodshed") as demo:
        gr.Markdown("# woodshed\nSong link -> stems -> sheet music you can read, plus key, Camelot and tempo.")
        with gr.Row():
            with gr.Column(scale=3):
                link = gr.Textbox(label="Song link (YouTube etc.) or path to a local file",
                                  placeholder="https://www.youtube.com/watch?v=...")
                upload = gr.Audio(label="...or upload / record", type="filepath",
                                  sources=["upload", "microphone"])
                with gr.Row():
                    start = gr.Textbox(label="Start (m:ss)", placeholder="0:30")
                    end = gr.Textbox(label="End (m:ss)", placeholder="1:30")
                gr.Markdown("_Tip: transcribe one section (the head, a solo) at a time - "
                            "faster, and easier to check by ear._")
            with gr.Column(scale=2):
                parts = gr.CheckboxGroup(PART_CHOICES, value=["other", "vocals", "bass"],
                                         label="Parts to transcribe")
                grid = gr.Radio(GRID_CHOICES, value="swing8", label="Rhythm grid")
                with gr.Row():
                    downbeat = gr.Slider(0, 3, value=0, step=1, label="Shift barline (beats)")
                    bpm = gr.Number(value=None, label="Tempo hint (BPM, blank = auto)")
                flute_fit = gr.Checkbox(value=True, label="Move parts into flute range (octaves only)")
                go = gr.Button("Transcribe", variant="primary")
        summary = gr.Markdown()
        sheets = gr.HTML()
        files = gr.File(label="Downloads (MusicXML opens in MuseScore; MIDI; Song JSON)",
                        file_count="multiple")
        with gr.Accordion("Stems (listen to check what each part contains)", open=False):
            with gr.Row():
                stem_audio = [gr.Audio(label=s, type="filepath", interactive=False)
                              for s in stems.STEMS]
        go.click(run, [link, upload, start, end, parts, grid, downbeat, bpm, flute_fit],
                 [summary, sheets, files, *stem_audio])
    return demo


if __name__ == "__main__":
    build().launch(allowed_paths=[str(fetch.RUNS)])
