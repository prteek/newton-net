"""Gradio app: ask Newton net a short driving-physics question and get the answer with a trace of how it was worked out.

The pipeline is decisions.md's: extractor → reader + router → assembler → planner → answer, all in model/. The reader
+ router is a keyword stand-in until the TinyLLM is trained.

Run from the repo root: uv run --with gradio==6.30.0 python space/app.py
"""
import sys
from pathlib import Path

import gradio as gr

sys.path.append(str(Path(__file__).resolve().parent.parent))  # model/ sits next to app.py on the Space, one up in the repo
from model.assembler import assemble
from model.extract import extract, mask
from model.planner import solve
from model.reader import read

CODE_URL = "https://github.com/prteek/newton-net"
MAX_CHARS = 160
DIAGRAM = Path(__file__).with_name("architecture.svg").read_text()
REFUSAL = (
    "**Can't answer this one.** I can only answer questions about the physics of motion, please check:\n"
    "- The question is about braking, accelerating or driving at a steady speed\n"
    "- Every value has a unit, e.g. speed in mph, km/h or m/s\n"
    "- It gives enough to work the answer out (e.g. the speed and the deceleration)"
)
TARGETS = {"v": "speed", "t": "time", "s": "distance", "a": "acceleration", "F": "force"}
SI_UNIT = {"u": "m/s", "v": "m/s", "a": "m/s²", "t": "s", "s": "m", "m": "kg", "F": "N"}


def number(x):
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def display(symbol, x):
    """Answer in display units: s or h; m or km; km/h; m/s²; N."""
    if symbol == "t":
        return f"{number(x / 3600)} h" if x > 3600 else f"{number(x)} s"
    if symbol == "s":
        return f"{number(x / 1000)} km" if x > 2000 else f"{number(x)} m"
    if symbol == "v":
        return f"{number(x * 3.6)} km/h"
    return f"{number(x).replace('-', '−')} {SI_UNIT[symbol]}"


def respond(question):
    """(answer markdown, trace markdown): each stage's output, as far as the pipeline got."""
    quantities = extract(question or "")
    marked = mask(question or "", quantities)
    trace = [f"**1. Extractor:** {marked}",
             "\n".join(f"- {q['surface']} → {q['dimension']} = {q['si']:.4g} SI" for q in quantities)]
    choice = read(marked)
    trace.append(f"**2. Reader + router** (keyword stand-in): {choice}")
    known = assemble(quantities, choice) if choice else None
    if known:
        trace.append("**3. Assembler** (SI): " + ", ".join(f"{k} = {v:.4g} {SI_UNIT[k]}" for k, v in known.items()))
    plan = solve(known, choice["target"]) if known else None
    if plan is None:
        return REFUSAL, "\n\n".join(trace)
    steps, values = plan
    target = choice["target"]
    trace += ["**4. Planner** (SI)", "\n".join(f"{i}. `{s}`" for i, s in enumerate(steps, 1)) or f"{target} is given"]
    return f"## {TARGETS[target].capitalize()}: {target} = {display(target, values[target])}", "\n\n".join(trace)


EXAMPLES = [
    "I'm doing 70 mph and want to slow to 10 mph in 5 s. How much braking force does that take?",
    "A car at 30 m/s brakes at 0.7g. How long until it stops?",
    "doing 70 mph in the rain, how far to stop?",
    "there's a dog 30 m ahead and I'm at 40 mph. how hard do I need to brake?",
    "A car at 50 mph brakes at 6 m/s² with a wall 30 m ahead. How fast does it hit the wall?",
    "How long does a car take to go from 30 mph to 60 mph at 3 m/s²?",
    "Pulling away from rest at 3 m/s², how far have I gone after 8 s?",
    "A 1,500 kg car goes from 30 mph to 60 mph in 6 s. What net force acts on it?",
    "cruising at 70 mph for 20 minutes, how far will I get?",
    "How long does it take to reach 60 mph?",
    "Is it legal to drive at 80 mph on a motorway?",
]

with gr.Blocks(title="Newton net") as demo:
    gr.HTML(
        '<div style="padding:10px 14px;border-radius:8px;border:1px solid #d97706;background:rgba(245,158,11,0.15)">'
        "<b>Demo only:</b> this is not the real model. Questions are read by keyword rules standing in for the "
        "TinyLLM, which hasn't been trained yet. Everything is a minimal first version, developed further as we go.</div>"
    )
    gr.Markdown(
        "# Newton net: exact driving physics from a question\n"
        "Ask one short question about braking, accelerating or cruising. A reader works out what is asked and what is "
        "known, and a fixed planner chains the equations of motion to an exact answer, with a trace of every step. "
        "Anything it can't work out is refused.\n\n"
        "**How it differs from a typical LLM**\n"
        "- The maths is exact: the network only reads the question; a fixed planner inside the same forward pass "
        "does the physics, so numbers are never guessed token by token, and no tool is called.\n"
        "- Every answer shows its working (values read, the reader's choices, each equation), and anything it can't "
        "work out is refused instead of answered anyway.\n"
        "- It is tiny (about 0.5M parameters, CPU only) because it only learns to read; the physics needs no training."
    )
    gr.HTML(DIAGRAM)
    gr.Markdown(
        "*The reader is a keyword placeholder until the TinyLLM is trained. Assumes dry/wet/icy braking at "
        "7.5/5.0/2.0 m/s², no reaction time, and a 2,000 kg car unless stated.*"
    )
    question = gr.Textbox(label=f"Your question (up to {MAX_CHARS} characters)", max_length=MAX_CHARS, lines=2,
                          placeholder="doing 70 mph in the rain, how far to stop?")
    gr.Examples(EXAMPLES, question, examples_per_page=len(EXAMPLES))
    ask = gr.Button("Answer", variant="primary")
    answer = gr.Markdown()
    with gr.Accordion("Trace", open=True):
        steps = gr.Markdown()
    gr.Markdown(f"Code: [{CODE_URL.removeprefix('https://')}]({CODE_URL}).")
    gr.on([ask.click, question.submit], respond, question, [answer, steps])

if __name__ == "__main__":
    demo.launch()
