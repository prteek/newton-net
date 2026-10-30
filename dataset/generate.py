"""Synthetic Newton net examples, labelled by the system requirements in decisions.md.

Run from the repo root: `uv run python -m dataset.generate --n 100 --out data/sample_100.jsonl`

Each question is labelled for the reader + router (target, stated values, implied values) and with
what the whole pipeline should do with it: answer, or refuse at the stage that can judge it. Answers
come from the planner (model/planner.py), so the data and the model share one source of physics.

Each line of the output is one JSON record:

- text: the question, at most 160 characters
- case: answer, or the refusal it should get: off_topic | ambiguous | multiple_questions | no_question
  (reader + router), duplicate (assembler), unknown_unit (unit converter), missing_input | conflict |
  impossible (planner)
- level: for answers, how the question is phrased: explicit | informal | implicit_unit | implied_value
- target, target_id: one of TARGETS; "refuse" only for the reader + router's own refusals. Questions the
  assembler, converter or planner refuse still carry the quantity they ask for
- motion: braking | accelerating | constant | null (metadata; the model only sees it through
  implied values and a_sign)
- surface: dry | wet | icy | null
- mentions: each value written in the text, in order: {quantity, value, unit, si, value_span,
  unit_span}. quantity is one of QUANTITIES; spans are [start, end) character offsets; si is null when
  the unit isn't one the converter knows. Accelerations and forces are written as magnitudes
- implied: {u, v, a, m}: values the question implies but doesn't state, or null. a is signed
- a_sign: -1 when braking, +1 when accelerating, null otherwise; applied to a stated a and F
- known_si: what the planner is given (stated values in SI, signed, plus implied values for
  quantities not stated); null when an earlier stage refuses
- outcome: answer | refuse; refused_by: reader | assembler | converter | planner | null; reason
- answer, answer_unit: the planner's answer in SI; null when refused
- plan: the planner's steps with their numbers; assumptions: implied values the plan used
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

from model.planner import solve, trace

TARGETS = ["speed", "time", "distance", "acceleration", "force", "refuse"]
TARGET_SYMBOL = {"speed": "v", "time": "t", "distance": "s", "acceleration": "a", "force": "F"}
TARGET_UNIT = {"speed": "m/s", "time": "s", "distance": "m", "acceleration": "m/s2", "force": "N"}
QUANTITIES = ["u", "v", "a", "t", "s", "m", "F"]
IMPLIED = ["u", "v", "a", "m"]
MOTIONS = ["braking", "accelerating", "constant"]
SURFACE_DECEL = {"dry": 7.5, "wet": 5.0, "icy": 2.0}
DEFAULT_MASS = 2000.0
MAX_CHARS = 160

# unit as written in text -> (factor to SI, decimals when written)
SPEED = {"m/s": (1.0, 0), "mps": (1.0, 0), "km/h": (1 / 3.6, 0), "kph": (1 / 3.6, 0), "mph": (0.44704, 0)}
UNITS = {
    "u": SPEED,
    "v": SPEED,
    "a": {"m/s2": (1.0, 1), "m/s²": (1.0, 1), "g": (9.80665, 2)},
    "t": {"s": (1.0, 0), "seconds": (1.0, 0), "min": (60.0, 0), "minutes": (60.0, 0), "hours": (3600.0, 1)},
    "s": {"m": (1.0, 0), "metres": (1.0, 0), "ft": (0.3048, 0), "km": (1000.0, 1), "miles": (1609.344, 1)},
    "m": {"kg": (1.0, 0), "tonnes": (1000.0, 1)},
    "F": {"N": (1.0, 0), "kN": (1000.0, 1)},
}
# units the converter doesn't know, for unknown_unit questions; factors only to write sensible values
UNKNOWN_UNITS = {
    "u": {"knots": (0.514444, 0), "ft/s": (0.3048, 0)},
    "s": {"yards": (0.9144, 0), "furlongs": (201.168, 1)},
    "a": {"ft/s²": (0.3048, 0)},
}
# units that read sensibly for a value of this size in SI: (quantity, upper bound) -> allowed units
SCALE = {
    ("t", 120): ["s", "seconds"],
    ("t", math.inf): ["min", "minutes", "hours"],
    ("s", 1500): ["m", "metres", "ft"],
    ("s", math.inf): ["km", "miles"],
}

# Placeholders: {u} … {F} write a value with a known unit; {u2} is a second, different value for u;
# {u_x}, {s_x}, {a_x} use a unit the converter doesn't know. {surface} is a road description and
# {rest} implies u = 0. The flag "stop" marks questions about coming to a stop, which imply v = 0.

ANSWERABLE = [  # (target, motion, level, template, flags)
    # braking
    ("distance", "braking", "explicit", "A car at {u} brakes at {a}. What is its stopping distance?", {"stop"}),
    ("distance", "braking", "explicit", "How far does a {m} car at {u} take to stop if it brakes at {a}?", {"stop"}),
    ("distance", "braking", "informal", "doing {u} and slam the brakes at {a}. how far till I stop?", {"stop"}),
    ("distance", "braking", "implicit_unit", "{u} braking at {a}. stopping distance?", {"stop"}),
    ("distance", "braking", "implied_value", "I'm doing {u} {surface}. How far will I travel before stopping?", {"stop"}),
    ("distance", "braking", "implied_value", "How far to stop from {u}?", {"stop"}),
    ("acceleration", "braking", "explicit", "A car at {u} must stop within {s}. What deceleration is needed?", {"stop"}),
    ("acceleration", "braking", "informal", "there's a dog {s} ahead and I'm at {u}. how hard do I need to brake?", {"stop"}),
    ("acceleration", "braking", "implicit_unit", "{u}, stop in {s}. decel needed?", {"stop"}),
    ("time", "braking", "explicit", "A car at {u} brakes at {a}. How long until it stops?", {"stop"}),
    ("time", "braking", "informal", "at {u}, braking at {a}, how long before I'm stationary?", {"stop"}),
    ("time", "braking", "implied_value", "How long does it take to stop from {u} {surface}?", {"stop"}),
    ("speed", "braking", "explicit", "A car at {u} sees an obstacle {s} ahead and brakes at {a}. How fast is it going when it reaches it?", set()),
    ("speed", "braking", "informal", "kid runs out {s} ahead, I'm at {u} and brake at {a}. how fast when I reach them?", set()),
    ("speed", "braking", "implied_value", "If I brake {surface} from {u} with a wall {s} ahead, how fast do I hit it?", set()),
    ("force", "braking", "explicit", "What braking force stops a {m} car from {u} within {s}?", {"stop"}),
    ("force", "braking", "informal", "{m} car braking at {a}, how much force is that?", set()),
    # accelerating
    ("distance", "accelerating", "explicit", "A car at {u} accelerates at {a} for {t}. How far does it travel?", set()),
    ("distance", "accelerating", "explicit", "How much distance does a car need to go from {u} to {v} accelerating at {a}?", set()),
    ("distance", "accelerating", "implied_value", "Pulling away {rest} at {a}, how far have I gone after {t}?", set()),
    ("distance", "accelerating", "informal", "merging onto the motorway, {u} up to {v} at {a}. how much slip road do I need?", set()),
    ("distance", "accelerating", "informal", "{F} of net thrust on a {m} car doing {u}, how far do I get in {t}?", set()),
    ("acceleration", "accelerating", "explicit", "What acceleration takes a car from {u} to {v} in {t}?", set()),
    ("acceleration", "accelerating", "implied_value", "My car does 0 to {v} in {t}. What's its average acceleration?", set()),
    ("acceleration", "accelerating", "informal", "need to get from {u} to {v} within {s} to overtake. how hard do I accelerate?", set()),
    ("time", "accelerating", "explicit", "How long does a car take to go from {u} to {v} at {a}?", set()),
    ("time", "accelerating", "implied_value", "Starting {rest}, how long to reach {v} at {a}?", set()),
    ("time", "accelerating", "implicit_unit", "{u} to {v} at {a}. time?", set()),
    ("speed", "accelerating", "explicit", "A car at {u} accelerates at {a} for {t}. What is its final speed?", set()),
    ("speed", "accelerating", "informal", "if I floor it from {u} at {a} for {s}, how fast will I be going?", set()),
    ("speed", "accelerating", "implied_value", "A {m} car pulls away {rest} at {a}. How fast is it after {t}?", set()),
    ("speed", "accelerating", "explicit", "A {m} car {rest} is pushed by a net force of {F}. How fast is it after {t}?", set()),
    ("force", "accelerating", "explicit", "What net force does a {m} car need to accelerate at {a}?", set()),
    ("force", "accelerating", "explicit", "A {m} car goes from {u} to {v} in {t}. What net force acts on it?", set()),
    ("force", "accelerating", "implied_value", "How much force does it take to get a car {rest} to {v} in {t}?", set()),
    # constant speed
    ("distance", "constant", "explicit", "A car travels at a constant {u} for {t}. How far does it go?", set()),
    ("distance", "constant", "informal", "cruising at {u} for {t}, how far will I get?", set()),
    ("distance", "constant", "implicit_unit", "{u} steady for {t}. distance?", set()),
    ("time", "constant", "explicit", "How long does it take to drive {s} at a constant {u}?", set()),
    ("time", "constant", "informal", "{s} to go and I'm cruising at {u}, how long till I get there?", set()),
    ("time", "constant", "implicit_unit", "{s} at {u}, steady. time?", set()),
]

# Refused by the reader + router: the target is "refuse"
READER_REFUSALS = {
    "off_topic": [
        "Why does my engine knock at {u}?",
        "How much fuel does a {m} car use at {u}?",
        "Is it legal to drive at {u} on a motorway?",
        "What tyre pressure should I use {surface}?",
        "What's the 0 to {v} time of a VW Golf GTI?",
        "Should I change gear before {u} when accelerating?",
        "What's the best car insurance for a new driver?",
        "Is it cheaper to charge an EV at home or at a supercharger?",
        "Can you write me a poem about a road trip?",
        "Which is safer {surface}, ABS or no ABS?",
        "What does a slow reaction time say about my reflexes?",
        "Will cruise control at {u} save me money on a {s} trip?",
        "How much horsepower does it take to produce {F} at the wheels?",
    ],
    "ambiguous": [
        "What about {a}?",
        "Is {u} too fast {surface}?",
        "{u} and {s}, what's the answer?",
        "Can I make it at {u}?",
        "How much is {t} at {u} braking?",
    ],
    "multiple_questions": [
        "How far and how long does it take to stop from {u} at {a}?",
        "From {u} at {a}, what's my stopping distance? And the force for a {m} car?",
        "How fast am I after {t} at {a} {rest}, and how far have I gone?",
        "Stopping {surface} from {u}: how long does it take and how far do I travel?",
    ],
    "no_question": [
        "I'm doing {u} {surface}.",
        "Accelerated from {u} to {v} in {t} on the way to work today.",
        "My {m} car brakes at {a}.",
        "Cruising at {u} on the motorway, {s} still to go.",
    ],
}

# Refused by a later stage: (case, target, motion, template, flags, how to make it fail)
STAGE_REFUSALS = [
    # assembler: two stated values for one quantity
    ("duplicate", "distance", "braking", "I'm doing {u} or maybe {u2}. How far to stop at {a}?", {"stop"}, None),
    ("duplicate", "time", "braking", "From {u}, no wait {u2}, how long to stop {surface}?", {"stop"}, None),
    ("duplicate", "distance", "constant", "Cruising at {u}, then {u2}, for {t}. How far do I go?", set(), None),
    # unit converter: a unit it doesn't know
    ("unknown_unit", "distance", "braking", "A car at {u_x} brakes at {a}. How far does it take to stop?", {"stop"}, None),
    ("unknown_unit", "time", "constant", "How long to cover {s_x} cruising at {u}?", set(), None),
    ("unknown_unit", "speed", "accelerating", "Accelerating {rest} at {a_x} for {t}, how fast am I going?", set(), None),
    # planner: an input is missing
    ("missing_input", "distance", "braking", "How far will I travel before stopping {surface}?", {"stop"}, None),
    ("missing_input", "distance", "constant", "How far do I get if I cruise for {t}?", set(), None),
    ("missing_input", "acceleration", "braking", "What deceleration do I need to stop within {s}?", {"stop"}, None),
    ("missing_input", "time", "accelerating", "How long does it take to reach {v}?", set(), None),
    ("missing_input", "time", "constant", "How long will it take me to cruise {s}?", set(), None),
    ("missing_input", "force", "accelerating", "What force does a {m} car need?", set(), None),
    # planner: stated values contradict each other (one value scaled away from the rest)
    ("conflict", "distance", "accelerating", "A car goes from {u} to {v} in {t}, accelerating at {a}. How far does it travel?", set(), "a"),
    ("conflict", "time", "accelerating", "A {m} car {rest} is pushed by {F} and accelerates at {a}. How long to reach {v}?", set(), "F"),
    ("conflict", "distance", "braking", "A car at {u} brakes at {a} and stops in {t}. How far does it go?", {"stop"}, "t"),
    # planner: the motion is impossible (it stops first)
    ("impossible", "speed", "braking", "A car at {u} brakes at {a}. How fast is it going when it reaches a wall {s} ahead?", set(), "s_past_stop"),
    ("impossible", "distance", "braking", "A car at {u} brakes at {a} for {t}. How far does it travel?", set(), "t_past_stop"),
]

SURFACE_PHRASES = {
    "dry": ["on a dry road", "on dry tarmac", "in good conditions"],
    "wet": ["in the rain", "on a wet road", "on wet tarmac"],
    "icy": ["on ice", "on an icy road", "in icy conditions"],
}
REST_PHRASES = ["from rest", "from a standstill", "from stationary"]

# share of each case in a generated set
MIX = {"answer": 0.60, "off_topic": 0.10, "ambiguous": 0.04, "multiple_questions": 0.04, "no_question": 0.04,
       "missing_input": 0.06, "conflict": 0.04, "impossible": 0.04, "duplicate": 0.02, "unknown_unit": 0.02}
EXPECTED_STAGE = {"duplicate": "assembler", "unknown_unit": "converter",
                  "missing_input": "planner", "conflict": "planner", "impossible": "planner"}


def sample_state(rng, motion, rest):
    """A consistent physical scenario in SI (magnitudes), from which a question picks its givens."""
    m = rng.uniform(900, 2200)
    if motion == "braking":
        u, a = rng.uniform(5, 35), rng.uniform(1, 9)
        s = rng.uniform(0.2, 1.0) * u**2 / (2 * a)
        return {"u": u, "a": a, "s": s, "t": rng.uniform(0.3, 1.0) * u / a, "v": 0.0, "m": m, "F": m * a}
    if motion == "accelerating":
        u, a = 0.0 if rest else rng.uniform(3, 25), rng.uniform(1, 5)
        t = rng.uniform(2, min(12, (45 - u) / a))  # top out at 45 m/s (about 100 mph)
        return {"u": u, "a": a, "t": t, "v": u + a * t, "s": u * t + a * t**2 / 2, "m": m, "F": m * a}
    u, t = rng.uniform(5, 35), rng.uniform(30, 7200)
    return {"u": u, "t": t, "s": u * t, "v": u, "a": 0.0, "m": m, "F": 0.0}


def as_written(rng, quantity, si, units=None):
    """Write an SI value in a unit that suits its size, then convert back, so labels match the text."""
    known = units is None
    units = UNITS[quantity] if known else units
    choices = next((u for (q, hi), u in SCALE.items() if known and q == quantity and si < hi), list(units))
    for _ in range(20):
        unit = rng.choice(choices)
        factor, decimals = units[unit]
        value = round(si / factor, decimals)
        if value > 0:
            value = int(value) if decimals == 0 else value
            return {"quantity": quantity, "value": value, "unit": unit, "si": round(value * factor, 6) if known else None}
    raise ValueError(f"cannot write {si} for {quantity}")


def quantity_of(placeholder):
    """{u2} and {u_x} are both values of u."""
    return placeholder.partition("_")[0].rstrip("2")


def render(rng, template, state, surface_phrase, rest_phrase):
    """Fill a template; return the text and its mentions with character spans."""
    text, pos, mentions = "", 0, []
    for match in re.finditer(r"\{(\w+)\}", template):
        text += template[pos:match.start()]
        name = match.group(1)
        if name == "surface":
            text += surface_phrase
        elif name == "rest":
            text += rest_phrase
        else:
            quantity = quantity_of(name)
            if name.endswith("2"):
                mention = as_written(rng, quantity, state[quantity] * rng.choice([0.6, 0.75, 1.3, 1.5]))
            elif name.endswith("_x"):
                mention = as_written(rng, quantity, state[quantity], UNKNOWN_UNITS[quantity])
            else:
                mention = as_written(rng, quantity, state[quantity])
            value = f"{mention['value']:,}" if mention["unit"] == "kg" else str(mention["value"])
            mention["value_span"] = [len(text), len(text) + len(value)]
            text += value + ("" if mention["unit"] == "g" else " ")
            mention["unit_span"] = [len(text), len(text) + len(mention["unit"])]
            text += mention["unit"]
            mentions.append(mention)
        pos = match.end()
    return text + template[pos:], mentions


def implied_values(target, motion, template, flags, surface, stated):
    """What the question implies but doesn't state (decisions.md, 2. Reader + router).

    The braking deceleration from the road surface is only implied when nothing else fixes it: not
    when the deceleration is what's asked, nor when the question says where the car must stop.
    """
    implied = dict.fromkeys(IMPLIED)
    if "{rest}" in template or "0 to {" in template:
        implied["u"] = 0.0
    if "stop" in flags:
        implied["v"] = 0.0
    if motion == "constant":
        implied["a"] = 0.0
    determined = target == "acceleration" or ("stop" in flags and "s" in stated)
    if motion == "braking" and not {"a", "F"} & stated and not determined:
        implied["a"] = -SURFACE_DECEL[surface or "dry"]
    implied["m"] = DEFAULT_MASS
    a_sign = {"braking": -1, "accelerating": 1}.get(motion)
    return {q: (None if q in stated else v) for q, v in implied.items()}, a_sign


def assemble(mentions, implied, a_sign):
    """Stages 3 and 4: one value per quantity, converted to SI. Returns (known, refused_by, reason)."""
    quantities = [m["quantity"] for m in mentions]
    if dupes := sorted({q for q in quantities if quantities.count(q) > 1}):
        return None, "assembler", f"two values given for {', '.join(dupes)}"
    if unknown := [m["unit"] for m in mentions if m["si"] is None]:
        return None, "converter", f"unit not recognised: {', '.join(unknown)}"
    known = {m["quantity"]: m["si"] * (a_sign or 1) if m["quantity"] in ("a", "F") else m["si"] for m in mentions}
    known.update({q: v for q, v in implied.items() if v is not None and q not in known})
    return known, None, None


def sensible(target, answer, known):
    """Keep answers in the range real driving questions have."""
    if target == "speed":
        return 0 < answer < 60 and (known.get("a", 0) >= 0 or answer < known["u"])
    if target == "time":
        return 0 < answer < 3 * 3600
    if target == "distance":
        return 0 < answer < 250_000
    if target == "acceleration":
        return 0.5 <= abs(answer) <= 10
    return 0.5 <= abs(answer) / known["m"] <= 10


def labelled(text, case, level, target, motion, surface, mentions, implied, a_sign):
    return {
        "text": text, "case": case, "level": level,
        "target": target, "target_id": TARGETS.index(target), "motion": motion, "surface": surface,
        "mentions": mentions, "implied": implied, "a_sign": a_sign,
        "known_si": None, "outcome": "refuse", "refused_by": None, "reason": None,
        "answer": None, "answer_unit": None, "plan": None, "assumptions": None,
    }


def through_pipeline(rec):
    """Run stages 3–5 on a labelled question and record what the pipeline does with it."""
    known, refused_by, reason = assemble(rec["mentions"], rec["implied"], rec["a_sign"])
    if refused_by:
        return rec | {"refused_by": refused_by, "reason": reason}
    symbol = TARGET_SYMBOL[rec["target"]]
    steps, values, reason = solve(known, symbol)
    rec = rec | {"known_si": known}
    if steps is None:
        return rec | {"refused_by": "planner", "reason": reason}
    used = {i for _, ins, *_ in steps for i in ins}
    return rec | {"outcome": "answer", "answer": round(values[symbol], 4), "answer_unit": TARGET_UNIT[rec["target"]],
                  "plan": trace(steps, values),
                  "assumptions": [q for q, v in rec["implied"].items() if v is not None and q in used]}


def planner_kind(reason):
    """Which planner refusal a reason describes: a conflict, a missing input, or a failed guard."""
    if reason.startswith("conflicting"):
        return "conflict"
    return "missing_input" if reason.startswith("missing") else "impossible"


def make_question(rng, case, target, motion, template, flags, level=None, fail=None):
    names = re.findall(r"\{(\w+)\}", template)
    rest = "rest" in names or "0 to {" in template
    surface = rng.choice(list(SURFACE_DECEL)) if "surface" in names else None
    surface_phrase = rng.choice(SURFACE_PHRASES[surface]) if surface else ""
    stated = {quantity_of(n) for n in names if n not in ("surface", "rest")}

    for _ in range(300):
        state = sample_state(rng, motion or "accelerating", rest)  # reader refusals: every value non-zero
        if surface and motion == "braking" and "a" not in stated:  # braking at the surface's deceleration
            state["a"] = SURFACE_DECEL[surface]
            state["s"] = rng.uniform(0.2, 1.0) * state["u"] ** 2 / (2 * state["a"])
        if fail in ("a", "F", "t"):
            state[fail] *= rng.choice([0.5, 0.6, 1.6, 2.0])
        elif fail == "s_past_stop":
            state["s"] = state["u"] ** 2 / (2 * state["a"]) * rng.uniform(1.3, 2.5)
        elif fail == "t_past_stop":
            state["t"] = state["u"] / state["a"] * rng.uniform(1.3, 2.5)
        text, mentions = render(rng, template, state, surface_phrase, rng.choice(REST_PHRASES))
        if len(text) > MAX_CHARS:
            continue

        if target == "refuse":  # the reader + router refuses; nothing is implied
            rec = labelled(text, case, case, "refuse", None, surface, mentions, dict.fromkeys(IMPLIED), None)
            return rec | {"refused_by": "reader", "reason": case}

        implied, a_sign = implied_values(target, motion, template, flags, surface, stated)
        rec = through_pipeline(labelled(text, case, level or case, target, motion, surface, mentions, implied, a_sign))
        if case == "answer":
            if rec["outcome"] == "answer" and sensible(target, rec["answer"], rec["known_si"]):
                return rec
        elif rec["refused_by"] == EXPECTED_STAGE[case] and (rec["refused_by"] != "planner" or planner_kind(rec["reason"]) == case):
            return rec
    raise RuntimeError(f"no valid {case} sample for: {template}")


def generate(n, seed=0):
    rng = random.Random(seed)
    counts = {case: round(share * n) for case, share in MIX.items()}
    counts["answer"] += n - sum(counts.values())
    answerable = rng.sample(ANSWERABLE, len(ANSWERABLE))
    records = []
    for i in range(counts["answer"]):
        target, motion, level, template, flags = answerable[i % len(answerable)]
        records.append(make_question(rng, "answer", target, motion, template, flags, level))
    for case, templates in READER_REFUSALS.items():
        records += [make_question(rng, case, "refuse", None, templates[i % len(templates)], set()) for i in range(counts[case])]
    for case in EXPECTED_STAGE:
        options = [r[1:] for r in STAGE_REFUSALS if r[0] == case]
        for i in range(counts[case]):
            target, motion, template, flags, fail = options[i % len(options)]
            records.append(make_question(rng, case, target, motion, template, flags, fail=fail))
    rng.shuffle(records)
    return [{"id": i, **r} for i, r in enumerate(records)]


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
