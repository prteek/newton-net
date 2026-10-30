"""Planner over the equations of motion: chains exact rules from the known quantities to the target.

Quantities: u (starting speed), v (final speed), a (acceleration, negative when braking), t (time),
s (distance), m (mass), F (force), all in SI. Values may be floats or torch tensors; rules are written
with plain arithmetic and `sqrt`, so gradients flow through a plan when the inputs are tensors.

    steps, values, reason = solve({"u": 20.0, "v": 0.0, "a": -5.0}, "s")

Each rule has a guard: a check on its inputs (and its result) that returns a reason when the rule
can't be applied, e.g. dividing by a zero time, or a car that stops before covering the distance.
A guarded rule is skipped, so another rule can still reach the target; if none can, the reasons
explain the refusal.

Zero and sign checks allow a small tolerance (ATOL, in SI), since values read by a trained model are
rarely exactly 0. When a question gives more values than the answer needs, every rule that can
recompute a value from the others is checked against it; a mismatch beyond REL_TOL is refused as a
conflict, e.g. "0 to 60 mph in 6 s at 2 m/s²" (6 s implies 4.5 m/s²).
"""

from collections import deque

import math

import torch

ATOL = 1e-3  # SI units: |x| below this counts as zero
REL_TOL = 0.05  # relative gap between a stated value and the value other inputs imply; allows for rounding
CONSISTENCY_ATOL = 0.05  # absolute gap allowed when both values are near zero


def sqrt(x):
    # clamp: within tolerance of zero, rounding can leave a tiny negative under the root
    return torch.sqrt(x.clamp(min=0)) if isinstance(x, torch.Tensor) else max(x, 0) ** 0.5


def val(x):
    return x.item() if isinstance(x, torch.Tensor) else x


def nonzero(name):
    return lambda q: f"{name} is 0" if abs(val(q[name])) < ATOL else None


def positive(name):
    return lambda q: f"{name} must be positive, got {val(q[name]):.4g}" if val(q[name]) < ATOL else None


def all_of(*checks):
    return lambda q: next((r for c in checks if (r := c(q)) is not None), None)


def reaches(q):
    # within REL_TOL of the stopping distance counts as reaching it (at v = 0): s is often rounded
    if val(q["u"]) ** 2 + 2 * val(q["a"]) * val(q["s"]) < -REL_TOL * val(q["u"]) ** 2 - ATOL:
        return f"it stops after {-val(q['u']) ** 2 / (2 * val(q['a'])):.4g} m, before covering {val(q['s']):.4g} m"
    return None


def no_reversal(q):
    if val(q["a"]) < -ATOL and val(q["t"]) > -val(q["u"]) / val(q["a"]) * (1 + REL_TOL) + ATOL:
        return f"it stops after {-val(q['u']) / val(q['a']):.4g} s, before {val(q['t']):.4g} s"
    return None


# (output, inputs, formula, guard on inputs, description). Rules from decisions.md (Equations), plus
# v = sqrt(u² + 2as), v = 2s/t - u and s = (u + v)/2 · t so that any three of u, v, a, t, s give the rest
RULES = [
    ("v", ("u", "a", "t"), lambda u, a, t: u + a * t,                        all_of(positive("t"), no_reversal),       "v = u + a·t"),
    ("v", ("u", "a", "s"), lambda u, a, s: sqrt(u**2 + 2 * a * s),           all_of(positive("s"), reaches),           "v = √(u² + 2as)"),
    ("v", ("u", "s", "t"), lambda u, s, t: 2 * s / t - u,                    positive("t"),                            "v = 2s/t - u"),
    ("a", ("F", "m"),      lambda F, m: F / m,                               positive("m"),                            "a = F/m"),
    ("a", ("u", "v", "t"), lambda u, v, t: (v - u) / t,                      positive("t"),                            "a = (v - u)/t"),
    ("a", ("u", "v", "s"), lambda u, v, s: (v**2 - u**2) / (2 * s),          positive("s"),                            "a = (v² - u²)/2s"),
    ("a", ("u", "s", "t"), lambda u, s, t: 2 * (s - u * t) / t**2,           positive("t"),                            "a = 2(s - u·t)/t²"),
    ("s", ("u", "a", "t"), lambda u, a, t: u * t + a * t**2 / 2,             all_of(positive("t"), no_reversal),       "s = u·t + a·t²/2"),
    ("s", ("u", "v", "a"), lambda u, v, a: (v**2 - u**2) / (2 * a),          nonzero("a"),                             "s = (v² - u²)/2a"),
    ("s", ("u", "v", "t"), lambda u, v, t: (u + v) / 2 * t,                  positive("t"),                            "s = (u + v)/2 · t"),
    ("t", ("u", "v", "a"), lambda u, v, a: (v - u) / a,                      nonzero("a"),                             "t = (v - u)/a"),
    ("t", ("u", "v", "s"), lambda u, v, s: 2 * s / (u + v),                  lambda q: "u + v is 0" if abs(val(q["u"]) + val(q["v"])) < ATOL else None, "t = 2s/(u + v)"),
    ("t", ("u", "a", "s"), lambda u, a, s: 2 * s / (u + sqrt(u**2 + 2 * a * s)),
                                                                             all_of(positive("s"), reaches,
                                                                                    lambda q: "u and a are 0" if abs(val(q["u"])) < ATOL and abs(val(q["a"])) < ATOL else None),
                                                                                                                       "t = 2s/(u + √(u² + 2as))"),
    ("F", ("m", "a"),      lambda m, a: m * a,                               positive("m"),                            "F = m·a"),
]

# results that can't be physical whatever the rule; checked after each step
RESULT_CHECKS = {
    "t": lambda x: f"time comes out as {val(x):.4g} s" if val(x) < ATOL else None,
    "s": lambda x: f"distance comes out as {val(x):.4g} m" if val(x) < -ATOL else None,
    "v": lambda x: f"final speed comes out as {val(x):.4g} m/s" if val(x) < -ATOL else None,
}


def applicable(rule, values):
    """The rule's result, or None if an input is unknown, its guard fails or the result is unphysical."""
    out, ins, formula, guard, text = rule
    if not set(ins) <= values.keys() or guard(values) is not None:
        return None
    result = formula(*(values[i] for i in ins))
    return None if RESULT_CHECKS.get(out, lambda x: None)(result) is not None else result


def conflicts(known, values):
    """Recompute every value that other values imply and compare; return the first mismatch, if any."""
    values = dict(values)
    while True:  # first derive everything the inputs determine, so all redundancy is exposed
        new = {r[0]: x for r in RULES if r[0] not in values and (x := applicable(r, values)) is not None}
        if not new:
            break
        values.update(new)
    for rule in RULES:
        out, ins, formula, guard, text = rule
        if out not in values or (implied := applicable(rule, values)) is None:
            continue
        stated, implied = val(values[out]), val(implied)
        if not math.isclose(stated, implied, rel_tol=REL_TOL, abs_tol=CONSISTENCY_ATOL):
            source = "given as" if out in known else "worked out as"
            return f"{out} is {source} {stated:.4g}, but {text} gives {implied:.4g}"
    return None


def solve(known, target):
    """Shortest chain of rules from `known` (name -> value) to `target`.

    Returns (steps, values, None) on success, where steps are the rules applied in order and values
    holds every quantity computed; or (None, None, reason) when no chain reaches the target or the
    inputs contradict each other.
    """
    if (reason := conflicts(known, known)) is not None:
        return None, None, f"conflicting inputs: {reason}"
    if target in known:
        return [], dict(known), None
    queue, seen, blocked = deque([(dict(known), [])]), {frozenset(known)}, []
    while queue:
        values, steps = queue.popleft()
        for rule in RULES:  # fixed order, so the same question always gets the same plan
            out, ins, formula, guard, text = rule
            if out in values or not set(ins) <= values.keys():
                continue
            if (reason := guard(values)) is not None:
                blocked.append(f"{text}: {reason}")
                continue
            result = formula(*(values[i] for i in ins))
            if (reason := RESULT_CHECKS.get(out, lambda x: None)(result)) is not None:
                blocked.append(f"{text}: {reason}")
                continue
            if out == target:
                return steps + [rule], {**values, out: result}, None
            key = frozenset(values) | {out}
            if key not in seen:
                seen.add(key)
                queue.append(({**values, out: result}, steps + [rule]))
    if blocked:
        return None, None, "; ".join(dict.fromkeys(blocked))
    return None, None, f"missing inputs: {missing(set(known), target)}"


def missing(known, target, depth=2):
    """What the user could add to reach the target, e.g. '{a} or {t, u} or {s, u}'.

    A missing quantity that other rules could produce is expanded into what those rules lack, so the
    answer names inputs the user can give rather than intermediate results.
    """
    reachable = set(known)
    while True:
        new = {out for out, ins, *_ in RULES if out not in reachable and set(ins) <= reachable}
        if not new:
            break
        reachable |= new
    options = set()
    for out, ins, *_ in RULES:
        if out != target:
            continue
        lacking = set(ins) - reachable
        options.add(frozenset(lacking))
        if depth > 1 and len(lacking) == 1:  # one intermediate short: say what would produce it
            (gap,) = lacking
            for o in filter(None, missing(reachable, gap, depth - 1).split(" or ")):
                if target not in (o := frozenset(o.strip("{}").split(", "))):  # skip circular routes
                    options.add(o)
    options = {o for o in options if o and not any(p < o for p in options if p)}  # drop supersets
    options = sorted({tuple(sorted(o)) for o in options}, key=lambda o: (len(o), o))
    return " or ".join("{" + ", ".join(o) + "}" for o in options)


def trace(steps, values):
    return [f"{text} = {val(values[out]):.4g}" for out, ins, formula, guard, text in steps]
