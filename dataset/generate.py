"""Synthetic driving questions with exact answers, for training and evaluating Newton net.

    uv run python -m dataset.generate --n 5000 --seed 0 --out data/train.jsonl

Each record: {"id", "text", "answer", "unit", "case"}.
- answer: in SI, signed (negative acceleration or force when braking); null when the question is refused
- unit: the SI unit of the quantity asked for (m/s, s, m, m/s2, N); null when off topic
- case: answer | off_topic | missing_input

A question is a template filled in from a random scenario. Values are written in units the extractor reads
(model/extract.py), and the answer is the planner's (model/planner.py) on the values as written, plus what the
question implies (a stop means v = 0, from rest u = 0, cruising a = 0, braking on a road its typical deceleration).
"""

import argparse
import json
import random
from pathlib import Path

from model.assembler import DEFAULT_MASS
from model.planner import solve
from model.reader import SURFACE_DECEL

UNIT = {"v": "m/s", "t": "s", "s": "m", "a": "m/s2", "F": "N"}
SURFACES = {"dry": "on a dry road", "wet": "in the rain", "icy": "on an icy road"}

# (target, motion, template, implied); {surface} brakes at that road's typical deceleration
ANSWERABLE = [
    ("s", "braking", "A car at {u} brakes at {a}. What is its stopping distance?", {"v": 0.0}),
    ("s", "braking", "doing {u} {surface}, how far to stop?", {"v": 0.0}),
    ("t", "braking", "A car at {u} brakes at {a}. How long until it stops?", {"v": 0.0}),
    ("a", "braking", "there's a dog {s} ahead and I'm at {u}. how hard do I need to brake?", {"v": 0.0}),
    ("v", "braking", "A car at {u} brakes at {a} with a wall {s} ahead. How fast does it hit the wall?", {}),
    ("F", "braking", "What force stops a {m} car from {u} in {t}?", {"v": 0.0}),
    ("t", "accelerating", "How long does a car take to go from {u} to {v} at {a}?", {}),
    ("v", "accelerating", "A car at {u} accelerates at {a} for {t}. What is its final speed?", {}),
    ("s", "accelerating", "Pulling away from rest at {a}, how far have I gone after {t}?", {"u": 0.0}),
    ("a", "accelerating", "What acceleration takes a car from {u} to {v} in {t}?", {}),
    ("F", "accelerating", "A {m} car goes from {u} to {v} in {t}. What net force acts on it?", {}),
    ("s", "constant", "cruising at {u} for {t}, how far will I get?", {"a": 0.0}),
    ("t", "constant", "How long does it take to drive {s} at a constant {u}?", {"a": 0.0}),
]
MISSING_INPUT = [  # a target, but not enough given to work it out
    ("t", "accelerating", "How long does it take to reach {v}?"),
    ("s", "constant", "How far do I get if I cruise for {t}?"),
    ("F", "accelerating", "What force does a {m} car need?"),
]
OFF_TOPIC = [
    "Is it legal to drive at {u} on a motorway?",
    "How much fuel does a {m} car use at {u}?",
    "What tyre pressure should I use {surface}?",
    "Can you write me a poem about a road trip?",
]
MIX = {"answer": 0.8, "missing_input": 0.1, "off_topic": 0.1}


def scenario(rng, motion, rest):
    """A consistent motion in SI (sizes; braking's sign is applied later)."""
    m = rng.uniform(900, 2200)
    if motion == "braking":
        u, a = rng.uniform(5, 35), rng.uniform(1, 9)
        return {"u": u, "a": a, "s": rng.uniform(0.2, 0.9) * u**2 / (2 * a), "t": rng.uniform(0.3, 1.0) * u / a, "m": m}
    if motion == "accelerating":
        u, a, t = 0.0 if rest else rng.uniform(3, 20), rng.uniform(1, 4), rng.uniform(2, 8)
        return {"u": u, "a": a, "t": t, "v": u + a * t, "s": u * t + a * t**2 / 2, "m": m}
    u, t = rng.uniform(5, 35), rng.uniform(30, 3600)
    return {"u": u, "t": t, "s": u * t, "m": m}


def written(rng, q, si):
    """A value as text in a unit the extractor reads, and the SI value that text means."""
    if q in ("u", "v"):
        unit, factor = rng.choice([("mph", 0.44704), ("km/h", 1 / 3.6), ("m/s", 1.0)])
    elif q == "a":
        unit, factor = rng.choice([("m/s²", 1.0), ("g", 9.80665)])
    elif q == "t":
        unit, factor = ("s", 1.0) if si < 120 else ("minutes", 60.0)
    elif q == "s":
        unit, factor = ("m", 1.0) if si < 2000 else ("km", 1000.0)
    else:
        return f"{round(si, -1):,.0f} kg", round(si, -1)
    value = round(si / factor, 2 if unit == "g" else 1)
    return f"{value:g}{'' if unit == 'g' else ' '}{unit}", value * factor


def fill(rng, template, state, surface):
    """The template's text and the known values it states, in SI."""
    text, known = template, {}
    for q in ("u", "v", "a", "t", "s", "m"):
        if "{" + q + "}" in text:
            phrase, known[q] = written(rng, q, state[q])
            text = text.replace("{" + q + "}", phrase)
    return text.replace("{surface}", SURFACES[surface]), known


def answered(rng):
    target, motion, template, implied = rng.choice(ANSWERABLE)
    surface = rng.choice(list(SURFACES))
    text, known = fill(rng, template, scenario(rng, motion, rest="u" in implied), surface)
    if motion == "braking" and "a" in known:
        known["a"] = -known["a"]
    if "{surface}" in template:
        known["a"] = -SURFACE_DECEL[surface]
    known = {"m": DEFAULT_MASS} | implied | known
    steps, values = solve(known, target)
    return {"text": text, "answer": round(float(values[target]), 4), "unit": UNIT[target], "case": "answer"}


def refused(rng, case):
    surface = rng.choice(list(SURFACES))
    if case == "missing_input":
        target, motion, template = rng.choice(MISSING_INPUT)
        text, _ = fill(rng, template, scenario(rng, motion, rest=False), surface)
        return {"text": text, "answer": None, "unit": UNIT[target], "case": case}
    text, _ = fill(rng, rng.choice(OFF_TOPIC), scenario(rng, "constant", rest=False), surface)
    return {"text": text, "answer": None, "unit": None, "case": case}


def generate(n, seed=0):
    rng = random.Random(seed)
    cases = rng.choices(list(MIX), weights=list(MIX.values()), k=n)
    return [{"id": i, **(answered(rng) if c == "answer" else refused(rng, c))} for i, c in enumerate(cases)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("data/sample_100.jsonl"))
    args = parser.parse_args()
    records = generate(args.n, args.seed)
    args.out.parent.mkdir(exist_ok=True)
    args.out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    print(f"wrote {len(records)} records to {args.out}")
