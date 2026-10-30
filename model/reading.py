"""Readings: every way the reader + router's choices can complete what the extractor found.

The extractor fixes each value's dimension, so the learned part only decides what the dimension
leaves open:

- order: which speed is the start (u) and which the final speed (v); a lone speed can be either
- sign: braking (-1) or accelerating (+1), applied to a stated a and F
- u0, v0: whether "from rest" (u = 0) or "stop" (v = 0) is implied
- a_source, when no a or F is stated: none, zero (constant speed) or learned (braking on a road
  surface: sign × the model's |a|)

Training scores every reading by whether the planner reproduces the labelled answer (or refuses,
for a question labelled without one), and raises the probability of the readings that do. A question
has at most 2 × 2 × 2 × 2 × 3 = 48 readings, usually far fewer.

The fixed rules of the assembler live here too: one value per dimension (two for speed), mass is
2,000 kg unless stated, and an implied value never fills the quantity being asked for.
"""

from itertools import product

DEFAULT_MASS = 2000.0
SYMBOL = {"acceleration": "a", "time": "t", "distance": "s", "mass": "m", "force": "F"}
SPEED_ORDERS = {0: [()], 1: [("u",), ("v",)], 2: [("u", "v"), ("v", "u")]}


def readings(quantities, target, a_learned=1.0):
    """Each (choice, known) the learned choices allow, or ([], (stage, reason)) if the values can't be used.

    quantities come from model.extract.extract; target is a planner symbol (v, t, s, a, F);
    a_learned is the model's |a| for a road surface (a float, or a tensor during training).
    """
    if problems := [q for q in quantities if q["problem"]]:
        return [], ("extractor", "; ".join(f"{q['surface']}: {q['problem']}" for q in problems))
    speeds = [q["si"] for q in quantities if q["dimension"] == "speed"]
    if len(speeds) > 2:
        return [], ("assembler", "more than two speeds given")
    base = {}
    for q in quantities:
        if q["dimension"] != "speed":
            if SYMBOL[q["dimension"]] in base:
                return [], ("assembler", f"two values given for {q['dimension']}")
            base[SYMBOL[q["dimension"]]] = q["si"]
    base.setdefault("m", DEFAULT_MASS)
    signed = "a" in base or "F" in base

    out = []
    for order, sign, u0, v0, a_source in product(SPEED_ORDERS[len(speeds)], (-1, 1), (False, True), (False, True),
                                                 ("none", "zero", "learned")):
        known = base | dict(zip(order, speeds))
        if (u0 and ("u" in known or target == "u")) or (v0 and ("v" in known or target == "v")):
            continue
        if a_source != "none" and (signed or target == "a"):
            continue
        if not (signed or a_source == "learned") and sign == 1:
            continue  # the sign changes nothing here; keep one reading
        if u0:
            known["u"] = 0.0
        if v0:
            known["v"] = 0.0
        if a_source == "zero":
            known["a"] = 0.0
        elif a_source == "learned":
            known["a"] = sign * a_learned
        for q in ("a", "F"):
            if q in base:
                known[q] = sign * abs(base[q])  # abs: a value written with a minus isn't flipped twice
        choice = {"order": order, "sign": sign if signed or a_source == "learned" else None,
                  "u0": u0, "v0": v0, "a_source": a_source}
        out.append((choice, known))
    return out, None
