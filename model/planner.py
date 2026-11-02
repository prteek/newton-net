"""4. Planner: chains the equations of motion from the known values to the one asked for.

Quantities, in SI: u (starting speed), v (final speed), a (acceleration, negative when braking), t (time),
s (distance), m (mass), F (force). Values may be floats or torch tensors, so gradients flow through a plan.

    solve({"u": 20.0, "v": 0.0, "a": -5.0}, "s")  # -> ['s = (v² - u²)/2a = 40'], {..., "s": 40.0}

Returns None when the target can't be reached from the known values.
"""

import math

import torch


def sqrt(x):
    return torch.sqrt(x) if isinstance(x, torch.Tensor) else math.sqrt(x) if x >= 0 else math.nan


# (result, inputs, formula, as written)
RULES = [
    ("v", ("u", "a", "t"), lambda u, a, t: u + a * t,               "v = u + a·t"),
    ("v", ("u", "a", "s"), lambda u, a, s: sqrt(u**2 + 2 * a * s),  "v = √(u² + 2as)"),
    ("a", ("F", "m"),      lambda F, m: F / m,                      "a = F/m"),
    ("a", ("u", "v", "t"), lambda u, v, t: (v - u) / t,             "a = (v - u)/t"),
    ("a", ("u", "v", "s"), lambda u, v, s: (v**2 - u**2) / (2 * s), "a = (v² - u²)/2s"),
    ("s", ("u", "a", "t"), lambda u, a, t: u * t + a * t**2 / 2,    "s = u·t + a·t²/2"),
    ("s", ("u", "v", "a"), lambda u, v, a: (v**2 - u**2) / (2 * a), "s = (v² - u²)/2a"),
    ("t", ("u", "v", "a"), lambda u, v, a: (v - u) / a,             "t = (v - u)/a"),
    ("t", ("u", "v", "s"), lambda u, v, s: 2 * s / (u + v),         "t = 2s/(u + v)"),
    ("F", ("m", "a"),      lambda m, a: m * a,                      "F = m·a"),
]


def solve(known, target):
    """Apply the first rule that can run (rules giving the target first) until the target is known.

    Returns (steps, values): each step as written with its result, and every value known at the end.
    """
    values, steps = dict(known), []
    while target not in values:
        for out, ins, formula, text in sorted(RULES, key=lambda r: r[0] != target):
            if out in values or any(i not in values for i in ins):
                continue
            try:
                result = formula(*(values[i] for i in ins))
            except ZeroDivisionError:
                continue
            if math.isfinite(float(result)):
                values[out] = result
                steps.append(f"{text} = {float(result):.4g}")
                break
        else:
            return None
    return steps, values
