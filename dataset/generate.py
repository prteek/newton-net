"""Synthetic Newton net training examples, labelled exactly by the physics.

Run from the repo root: `uv run python -m dataset.generate --n 100 --out data/sample_100.jsonl`

Questions cover three motion types: braking, accelerating and constant speed. Each line of the output
is one JSON record:

- text: the question, at most 160 characters
- level: explicit | informal | implicit_unit | implied_value | insufficient | out_of_scope
- route, route_id: one of ROUTES
- motion: braking | accelerating | constant | null
- action, action_id: answer | refuse
- surface: dry | wet | icy | null
- slots: for each of SLOTS, null or {value, unit, si, value_span, unit_span}, where the spans are
  [start, end) character offsets into text, so slot labels can be projected onto any tokeniser.
  acceleration is written as a magnitude; its sign comes from the motion type
- inputs_si: the values the engine runs on, after unit conversion and defaults; null when refused.
  acceleration is signed (negative when braking)
- defaults: inputs not written in the text and where their value came from, e.g.
  {"acceleration": "surface", "speed": "from_rest", "mass": "default"}; a force question with no mass
  uses DEFAULT_MASS, and a force given instead of an acceleration gives a = F/m
- accel_bucket: index into ACCEL_BUCKETS (1.0 to 10.0 m/s² in 0.5 steps) of the acceleration
  magnitude, stated, defaulted or solved for; null at constant speed
- target, target_name, target_unit: the exact answer in SI; null when refused
- trace: the formula used and intermediate quantities
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

ROUTES = ["resulting_distance", "required_acceleration", "resulting_time", "resulting_speed", "required_force", "unsupported"]
MOTIONS = ["braking", "accelerating", "constant"]
ACTIONS = ["answer", "refuse"]
SLOTS = ["speed", "final_speed", "acceleration", "time", "distance", "mass", "force"]
ACCEL_BUCKETS = [1.0 + 0.5 * i for i in range(19)]
SURFACE_DECEL = {"dry": 7.5, "wet": 5.0, "icy": 2.0}
DEFAULT_MASS = 2000.0  # kg, used when a force question gives no mass
MAX_CHARS = 160

# unit as written in text -> (factor to SI, decimals when written)
SPEED = {"m/s": (1.0, 0), "mps": (1.0, 0), "km/h": (1 / 3.6, 0), "kph": (1 / 3.6, 0), "mph": (0.44704, 0)}
UNITS = {
    "speed": SPEED,
    "final_speed": SPEED,
    "acceleration": {"m/s2": (1.0, 1), "m/s²": (1.0, 1), "g": (9.80665, 2)},
    "time": {"s": (1.0, 0), "seconds": (1.0, 0), "min": (60.0, 0), "minutes": (60.0, 0), "hours": (3600.0, 1)},
    "distance": {"m": (1.0, 0), "metres": (1.0, 0), "ft": (0.3048, 0), "km": (1000.0, 1), "miles": (1609.344, 1)},
    "mass": {"kg": (1.0, 0), "tonnes": (1000.0, 1)},
    "force": {"N": (1.0, 0), "kN": (1000.0, 1)},
}
# units that read sensibly for a value of this size in SI: (slot, upper bound) -> allowed units
SCALE = {
    ("time", 120): ["s", "seconds"],
    ("time", math.inf): ["min", "minutes", "hours"],
    ("distance", 1500): ["m", "metres", "ft"],
    ("distance", math.inf): ["km", "miles"],
}

# (route, motion, level, template). {slot} expands to value + unit, {surface} to a road description,
# {rest} to a phrase that implies a starting speed of 0
TEMPLATES = [
    # braking
    ("resulting_distance", "braking", "explicit", "A car at {speed} brakes at {acceleration}. What is its stopping distance?"),
    ("resulting_distance", "braking", "explicit", "How far does a {mass} car at {speed} take to stop if it brakes at {acceleration}?"),
    ("resulting_distance", "braking", "informal", "doing {speed} and slam the brakes at {acceleration}. how far till I stop?"),
    ("resulting_distance", "braking", "implicit_unit", "{speed} braking at {acceleration}. stopping distance?"),
    ("resulting_distance", "braking", "implied_value", "I'm doing {speed} {surface}. How far will I travel before stopping?"),
    ("resulting_distance", "braking", "implied_value", "How far to stop from {speed} {surface}?"),
    ("required_acceleration", "braking", "explicit", "A car at {speed} must stop within {distance}. What deceleration is needed?"),
    ("required_acceleration", "braking", "informal", "there's a dog {distance} ahead and I'm at {speed}. how hard do I need to brake?"),
    ("required_acceleration", "braking", "implicit_unit", "{speed}, stop in {distance}. decel needed?"),
    ("resulting_time", "braking", "explicit", "A car at {speed} brakes at {acceleration}. How long until it stops?"),
    ("resulting_time", "braking", "informal", "at {speed}, braking at {acceleration}, how long before I'm stationary?"),
    ("resulting_time", "braking", "implied_value", "How long does it take to stop from {speed} {surface}?"),
    ("resulting_speed", "braking", "explicit", "A car at {speed} sees an obstacle {distance} ahead and brakes at {acceleration}. How fast is it going when it reaches it?"),
    ("resulting_speed", "braking", "informal", "kid runs out {distance} ahead, I'm at {speed} and brake at {acceleration}. how fast when I reach them?"),
    ("resulting_speed", "braking", "implied_value", "If I brake {surface} from {speed} with a wall {distance} ahead, how fast do I hit it?"),
    # accelerating
    ("resulting_distance", "accelerating", "explicit", "A car at {speed} accelerates at {acceleration} for {time}. How far does it travel?"),
    ("resulting_distance", "accelerating", "explicit", "How much distance does a car need to go from {speed} to {final_speed} accelerating at {acceleration}?"),
    ("resulting_distance", "accelerating", "implied_value", "Pulling away {rest} at {acceleration}, how far have I gone after {time}?"),
    ("resulting_distance", "accelerating", "informal", "merging onto the motorway, {speed} up to {final_speed} at {acceleration}. how much slip road do I need?"),
    ("required_acceleration", "accelerating", "explicit", "What acceleration takes a car from {speed} to {final_speed} in {time}?"),
    ("required_acceleration", "accelerating", "implied_value", "My car does 0 to {final_speed} in {time}. What's its average acceleration?"),
    ("required_acceleration", "accelerating", "informal", "need to get from {speed} to {final_speed} within {distance} to overtake. how hard do I accelerate?"),
    ("resulting_time", "accelerating", "explicit", "How long does a car take to go from {speed} to {final_speed} at {acceleration}?"),
    ("resulting_time", "accelerating", "implied_value", "Starting {rest}, how long to reach {final_speed} at {acceleration}?"),
    ("resulting_time", "accelerating", "implicit_unit", "{speed} to {final_speed} at {acceleration}. time?"),
    ("resulting_speed", "accelerating", "explicit", "A car at {speed} accelerates at {acceleration} for {time}. What is its final speed?"),
    ("resulting_speed", "accelerating", "informal", "if I floor it from {speed} at {acceleration} for {distance}, how fast will I be going?"),
    ("resulting_speed", "accelerating", "implied_value", "A {mass} car pulls away {rest} at {acceleration}. How fast is it after {time}?"),
    ("resulting_speed", "accelerating", "explicit", "A {mass} car {rest} is pushed by a net force of {force}. How fast is it after {time}?"),
    ("resulting_distance", "accelerating", "informal", "{force} of net thrust on a {mass} car doing {speed}, how far do I get in {time}?"),
    # force asked for (F = m·a, negative when braking); no mass stated means DEFAULT_MASS
    ("required_force", "accelerating", "explicit", "What net force does a {mass} car need to accelerate at {acceleration}?"),
    ("required_force", "accelerating", "explicit", "A {mass} car goes from {speed} to {final_speed} in {time}. What net force acts on it?"),
    ("required_force", "accelerating", "implied_value", "How much force does it take to get a car {rest} to {final_speed} in {time}?"),
    ("required_force", "braking", "explicit", "What braking force stops a {mass} car from {speed} within {distance}?"),
    ("required_force", "braking", "informal", "{mass} car braking at {acceleration}, how much force is that?"),
    # constant speed
    ("resulting_distance", "constant", "explicit", "A car travels at a constant {speed} for {time}. How far does it go?"),
    ("resulting_distance", "constant", "informal", "cruising at {speed} for {time}, how far will I get?"),
    ("resulting_distance", "constant", "implicit_unit", "{speed} steady for {time}. distance?"),
    ("resulting_time", "constant", "explicit", "How long does it take to drive {distance} at a constant {speed}?"),
    ("resulting_time", "constant", "informal", "{distance} to go and I'm sitting at {speed}, how long till I get there?"),
    ("resulting_time", "constant", "implicit_unit", "{distance} at {speed}, steady. time?"),
]

INSUFFICIENT = [  # in-domain questions missing an input nothing can fill in: route known, nothing to compute
    ("resulting_distance", "braking", "How far will I travel before stopping {surface}?"),
    ("resulting_distance", "constant", "How far do I get if I drive for {time}?"),
    ("required_acceleration", "braking", "What deceleration do I need to stop within {distance}?"),
    ("resulting_time", "accelerating", "How long does it take to reach {final_speed}?"),
    ("resulting_time", "constant", "How long will it take me to drive {distance}?"),
    ("resulting_speed", "accelerating", "How fast will I be going after accelerating for {time}?"),
    ("required_force", "accelerating", "What force does a {mass} car need?"),
]

OUT_OF_SCOPE = [  # near misses: driving words and numbers, but no route can answer them
    "Why does my engine knock at {speed}?",
    "How much fuel does a {mass} car use at {speed}?",
    "Is it legal to drive at {speed} on a motorway?",
    "What tyre pressure should I use {surface}?",
    "What's the 0 to {final_speed} time of a VW Golf GTI?",
    "Should I change gear before {speed} when accelerating?",
    "What's the best car insurance for a new driver?",
    "Is it cheaper to charge an EV at home or at a supercharger?",
    "Can you write me a poem about a road trip?",
    "Which is safer {surface}, ABS or no ABS?",
    "What does a slow reaction time say about my reflexes?",
    "Will cruise control at {speed} save me money on a {distance} trip?",
    "How much horsepower does it take to produce {force} at the wheels?",
]

SURFACE_PHRASES = {
    "dry": ["on a dry road", "on dry tarmac", "in good conditions"],
    "wet": ["in the rain", "on a wet road", "on wet tarmac"],
    "icy": ["on ice", "on an icy road", "in icy conditions"],
}
REST_PHRASES = ["from rest", "from a standstill", "from stationary"]


def sample_state(rng, motion, rest):
    """A consistent physical scenario in SI, from which a question picks its givens and its unknown."""
    if motion == "braking":
        v, a = rng.uniform(5, 35), rng.uniform(1, 9)
        return {"speed": v, "acceleration": a, "distance": rng.uniform(0.2, 1.0) * v**2 / (2 * a)}
    if motion == "accelerating":
        v, a = 0.0 if rest else rng.uniform(3, 25), rng.uniform(1, 5)
        t = rng.uniform(2, min(12, (45 - v) / a))  # top out at 45 m/s (about 100 mph)
        return {"speed": v, "acceleration": a, "time": t, "final_speed": v + a * t, "distance": v * t + a * t**2 / 2}
    v, t = rng.uniform(5, 35), rng.uniform(30, 7200)
    return {"speed": v, "time": t, "distance": v * t}


def as_written(rng, slot, si):
    """Write an SI value in a unit that suits its size, then convert back, so labels match the text."""
    units = next((u for (s, hi), u in SCALE.items() if s == slot and si < hi), list(UNITS[slot]))
    for _ in range(20):
        unit = rng.choice(units)
        factor, decimals = UNITS[slot][unit]
        value = round(si / factor, decimals)
        if value > 0:
            value = int(value) if decimals == 0 else value
            return {"value": value, "unit": unit, "si": round(value * factor, 6)}
    raise ValueError(f"cannot write {si} for {slot}")


def render(template, slots, surface_phrase, rest_phrase):
    """Fill a template and record character spans of each slot's value and unit."""
    text, pos = "", 0
    for m in re.finditer(r"\{(\w+)\}", template):
        text += template[pos:m.start()]
        name = m.group(1)
        if name == "surface":
            text += surface_phrase
        elif name == "rest":
            text += rest_phrase
        else:
            s = slots[name]
            value = f"{s['value']:,}" if s["unit"] == "kg" else str(s["value"])
            s["value_span"] = [len(text), len(text) + len(value)]
            text += value + ("" if s["unit"] in ("g", "ms") else " ")
            s["unit_span"] = [len(text), len(text) + len(s["unit"])]
            text += s["unit"]
        pos = m.end()
    return text + template[pos:]


def physics(route, motion, x):
    """The NewtonPhysics engine: exact answers in SI with a trace. x holds SI inputs; a is signed."""
    v, a = x.get("speed"), x.get("acceleration")
    if route == "required_force":
        a_used = a if a is not None else physics("required_acceleration", motion, x)[0]
        return x["mass"] * a_used, "force", "N", {"formula": "F = m*a", "acceleration": a_used}
    if motion == "braking":
        if route == "resulting_distance":
            return v**2 / (-2 * a), "stopping_distance", "m", {"formula": "v^2/(2|a|)"}
        if route == "required_acceleration":
            return -(v**2) / (2 * x["distance"]), "required_acceleration", "m/s2", {"formula": "-v^2/(2d)"}
        if route == "resulting_time":
            return v / -a, "stopping_time", "s", {"formula": "v/|a|"}
        return math.sqrt(max(v**2 + 2 * a * x["distance"], 0.0)), "impact_speed", "m/s", {"formula": "sqrt(v^2 - 2|a|d)"}
    if motion == "accelerating":
        t, vf, d = x.get("time"), x.get("final_speed"), x.get("distance")
        if route == "resulting_distance":
            if t is not None:
                return v * t + a * t**2 / 2, "distance", "m", {"formula": "v*t + a*t^2/2"}
            return (vf**2 - v**2) / (2 * a), "distance", "m", {"formula": "(vf^2 - v^2)/(2a)"}
        if route == "required_acceleration":
            if t is not None:
                return (vf - v) / t, "required_acceleration", "m/s2", {"formula": "(vf - v)/t"}
            return (vf**2 - v**2) / (2 * d), "required_acceleration", "m/s2", {"formula": "(vf^2 - v^2)/(2d)"}
        if route == "resulting_time":
            return (vf - v) / a, "time", "s", {"formula": "(vf - v)/a"}
        if t is not None:
            return v + a * t, "final_speed", "m/s", {"formula": "v + a*t"}
        return math.sqrt(v**2 + 2 * a * d), "final_speed", "m/s", {"formula": "sqrt(v^2 + 2ad)"}
    if route == "resulting_distance":
        return v * x["time"], "distance", "m", {"formula": "v*t"}
    return x["distance"] / v, "time", "s", {"formula": "d/v"}


def bucket(a):
    return min(range(len(ACCEL_BUCKETS)), key=lambda i: abs(ACCEL_BUCKETS[i] - abs(a)))


def base_record(text, level, route, motion, action, surface, slots):
    return {
        "text": text, "level": level,
        "route": route, "route_id": ROUTES.index(route), "motion": motion,
        "action": action, "action_id": ACTIONS.index(action),
        "surface": surface,
        "slots": {k: slots.get(k) for k in SLOTS},
        "inputs_si": None, "defaults": None, "accel_bucket": None,
        "target": None, "target_name": None, "target_unit": None, "trace": None,
    }


def valid(route, motion, target, x, text):
    signed = route in ("required_acceleration", "required_force")
    if len(text) > MAX_CHARS or not math.isfinite(target) or target <= 0 and not signed:
        return False
    if route == "required_acceleration":
        return 1.0 <= abs(target) <= 10.0 and (target < 0) == (motion == "braking")
    if route == "required_force":
        return 1.0 <= abs(target) / x["mass"] <= 10.0 and (target < 0) == (motion == "braking")
    if motion == "braking" and route == "resulting_speed":
        return 0 < target < x["speed"]  # it slows, but not to a stop
    if motion == "accelerating":
        return target < 2000 if route == "resulting_distance" else target < 60
    return True


def make_answerable(rng, route, motion, level, template):
    names = set(re.findall(r"\{(\w+)\}", template))
    rest = "rest" in names or "0 to {" in template  # "0 to 60 mph" also implies rest
    surface = rng.choice(list(SURFACE_DECEL)) if "surface" in names else None
    surface_phrase = rng.choice(SURFACE_PHRASES[surface]) if surface else ""
    rest_phrase = rng.choice(REST_PHRASES)

    for _ in range(200):
        state = sample_state(rng, motion, rest)
        if surface:  # braking on a described road, with no deceleration stated
            state["acceleration"] = SURFACE_DECEL[surface]
        if "mass" in names:
            state["mass"] = rng.uniform(900, 2200)
        if "force" in names:  # force given instead of acceleration
            state["force"] = state["mass"] * state["acceleration"]
        slots = {n: as_written(rng, n, state[n]) for n in names & set(SLOTS)}
        text = render(template, slots, surface_phrase, rest_phrase)

        defaults = {}
        x = {k: s["si"] for k, s in slots.items()}
        if rest:
            x["speed"], defaults["speed"] = 0.0, "from_rest"
        if surface:
            x["acceleration"], defaults["acceleration"] = SURFACE_DECEL[surface], "surface"
        if motion == "braking":
            if "acceleration" in x:
                x["acceleration"] = -x["acceleration"]
        if route == "required_force" and "mass" not in x:
            x["mass"], defaults["mass"] = DEFAULT_MASS, "default"
        if "force" in x and "acceleration" not in x:  # a = F/m, not stated in the text
            x["acceleration"] = x["force"] / x["mass"]
        target, target_name, target_unit, trace = physics(route, motion, x)
        if valid(route, motion, target, x, text):
            break
    else:
        raise RuntimeError(f"no valid sample for {route}/{motion}: {template}")

    record = base_record(text, level, route, motion, "answer", surface, slots)
    used_a = target if route == "required_acceleration" else trace.get("acceleration", x.get("acceleration"))
    record.update(
        inputs_si={k: x.get(k) for k in SLOTS},
        defaults=defaults,
        accel_bucket=bucket(used_a) if used_a is not None else None,
        target=round(target, 4), target_name=target_name, target_unit=target_unit,
        trace={k: round(val, 4) if isinstance(val, float) else val for k, val in trace.items()},
    )
    return record


def make_refused(rng, route, motion, template, level):
    names = set(re.findall(r"\{(\w+)\}", template))
    surface = rng.choice(list(SURFACE_DECEL)) if "surface" in names else None
    surface_phrase = rng.choice(SURFACE_PHRASES[surface]) if surface else ""
    state = {**sample_state(rng, "braking", False), **sample_state(rng, "accelerating", False),
             "mass": rng.uniform(900, 2200), "force": rng.uniform(1000, 10000)}
    if motion == "constant":
        state.update(sample_state(rng, "constant", False))
    slots = {n: as_written(rng, n, state[n]) for n in names & set(SLOTS)}
    return base_record(render(template, slots, surface_phrase, ""), level, route, motion, "refuse", surface, slots)


def generate(n, seed=0):
    """About 78% answerable across the three motion types, 8% in-domain but missing an input, 14% out of scope."""
    rng = random.Random(seed)
    n_refused_in, n_out = round(0.08 * n), round(0.14 * n)
    n_answer = n - n_refused_in - n_out
    order = rng.sample(TEMPLATES, len(TEMPLATES))
    records = [make_answerable(rng, *order[i % len(order)]) for i in range(n_answer)]
    records += [make_refused(rng, *INSUFFICIENT[i % len(INSUFFICIENT)], "insufficient") for i in range(n_refused_in)]
    records += [make_refused(rng, "unsupported", None, OUT_OF_SCOPE[i % len(OUT_OF_SCOPE)], "out_of_scope") for i in range(n_out)]
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
