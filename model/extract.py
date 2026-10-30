"""Fixed extractor: finds each number + unit in a question and converts it to SI.

quantulum3 finds the quantities (including number words like "seventy"); pint converts them and
gives their dimension. Nothing here is learned. Which role a value plays (start or final speed) and
its sign are left to the reader + router.

    extract("brakes from 70mph at 0.69g")
    -> [{"surface": "70mph", "dimension": "speed", "si": 31.29, "span": (12, 17), "problem": None}, ...]

A quantity the system can't use is kept with a `problem` so the pipeline can refuse with a reason:
a number with no unit ("doing 70"), or a unit of a dimension the physics doesn't cover ("150 hp").
"""

import re
import warnings

import pint

with warnings.catch_warnings():  # quantulum3 warns that its optional classifier isn't installed
    warnings.simplefilter("ignore")
    from quantulum3 import parser

UREG = pint.UnitRegistry()

DIMENSIONS = {
    "speed": UREG.get_dimensionality("m/s"),
    "acceleration": UREG.get_dimensionality("m/s**2"),
    "time": UREG.get_dimensionality("s"),
    "distance": UREG.get_dimensionality("m"),
    "mass": UREG.get_dimensionality("kg"),
    "force": UREG.get_dimensionality("N"),
}

# quantulum3 readings that are wrong for driving questions
UNIT_FIXES = {
    "gram": "standard_gravity",  # "0.69g" is braking in g, not a mass in grams
    "second of arc": "second",  # "6 seconds" is a time, not an angle
    "minute of arc": "minute",  # likewise "34 minutes"
}

# "to" or a dash between two values: "15 m/s to 45 mph", "0-60 mph", "30 to 10 m/s"
CONNECTOR = re.compile(r"(?<=[\w/²])(\s+to\s+|\s*[-–]\s*)(?=\d)")


def to_si(value, unit_name):
    """(dimension, SI value) for a quantulum3 unit name, or (None, None) if pint can't use it."""
    name = UNIT_FIXES.get(unit_name, unit_name).replace("square second", "second**2").replace(" per ", " / ")
    try:
        quantity = UREG.Quantity(value, name).to_base_units()
    except (pint.errors.UndefinedUnitError, pint.errors.DefinitionSyntaxError, AttributeError, TypeError):
        return None, None
    dimension = next((d for d, dim in DIMENSIONS.items() if quantity.dimensionality == dim), None)
    return dimension, float(quantity.magnitude) if dimension else None


def _parse(text):
    """quantulum3 quantities as (surface, value, unit name, span), each with a usable unit if there is one.

    quantulum3 sometimes swallows the next word into the unit ("27 km/h at" becomes kilometres per hour
    per ampere-turn); when a unit isn't usable, the last word is dropped and the rest parsed again.
    """
    out = []
    for q in parser.parse(text):
        surface, value, unit, span = text[q.span[0]:q.span[1]], q.value, q.unit.name, q.span
        while unit != "dimensionless" and not to_si(value, unit)[0] and " " in surface:
            retry = parser.parse(surface.rsplit(" ", 1)[0])
            if len(retry) != 1 or retry[0].unit.name == "dimensionless":
                break
            surface, value, unit = retry[0].surface, retry[0].value, retry[0].unit.name
            span = (span[0], span[0] + len(surface))
        out.append((surface, value, unit, span))
    return out


def extract(text):
    """Every number + unit in the text, with its dimension and SI value, in order of appearance.

    quantulum3 merges a range like "15 m/s to 45 m/s" into its mean, and drops a falling one ("30 to
    10 m/s") altogether. So each connector is replaced by ";" plus spaces before parsing (keeping
    every span in place), and a bare number just before a connector takes the next value's unit:
    "0 to 60 mph" gives 0 mph and 60 mph.
    """
    connectors = set()

    def mark(match):
        connectors.add(match.start())
        return ";" + " " * (len(match.group(0)) - 1)

    parsed = _parse(CONNECTOR.sub(mark, text))
    out = []
    for i, (surface, value, unit, span) in enumerate(parsed):
        surface = text[span[0]:span[1]]
        if unit == "dimensionless" and span[1] in connectors and i + 1 < len(parsed):
            unit = parsed[i + 1][2]
        if unit == "dimensionless":
            out.append({"surface": surface, "value": value, "unit": None, "dimension": None, "si": None,
                        "span": span, "problem": "number without a unit"})
            continue
        dimension, si = to_si(value, unit)
        out.append({"surface": surface, "value": value, "unit": unit, "dimension": dimension, "si": si,
                    "span": span, "problem": None if dimension else f"unit not usable: {unit}"})
    return out


def mask(text, quantities):
    """The question as the encoder sees it: each value replaced by a marker of its dimension.

    "A car at 15 m/s brakes at 0.69g" → "A car at <speed1> brakes at <acceleration>". Speeds are
    numbered in order, since the order head reads <speed1> and <speed2>; a value the extractor can't
    use becomes <number>.
    """
    out, pos, speeds = "", 0, 0
    for q in quantities:
        start, end = q["span"]
        if q["dimension"] == "speed":
            speeds += 1
            marker = f"<speed{speeds}>"
        else:
            marker = f"<{q['dimension'] or 'number'}>"
        out += text[pos:start] + marker
        pos = end
    return out + text[pos:]
