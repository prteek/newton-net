"""1. Extractor: finds each number + unit in a question and converts it to SI, with a regex.

    extract("A car at 70 mph brakes at 0.7g")
    -> [{"surface": "70 mph", "dimension": "speed", "si": 31.29, "span": (9, 15)}, ...]

Only the units in UNITS are read; a number without one of them is ignored.
"""

import re

# unit as written -> (dimension, factor to SI)
UNITS = {
    "mph": ("speed", 0.44704), "km/h": ("speed", 1 / 3.6), "kph": ("speed", 1 / 3.6), "m/s": ("speed", 1.0),
    "m/s²": ("acceleration", 1.0), "m/s2": ("acceleration", 1.0), "g": ("acceleration", 9.80665),
    "s": ("time", 1.0), "seconds": ("time", 1.0), "min": ("time", 60.0), "minutes": ("time", 60.0),
    "hours": ("time", 3600.0),
    "m": ("distance", 1.0), "metres": ("distance", 1.0), "km": ("distance", 1000.0), "miles": ("distance", 1609.344),
    "kg": ("mass", 1.0), "tonnes": ("mass", 1000.0),
    "N": ("force", 1.0), "kN": ("force", 1000.0),
}
# longest unit first, so "m/s²" wins over "m/s" and "m"; the unit must end the word
QUANTITY = re.compile(r"(\d[\d,]*(?:\.\d+)?)\s*(" + "|".join(re.escape(u) for u in sorted(UNITS, key=len, reverse=True))
                      + r")(?![\w/²])")


def extract(text):
    out = []
    for match in QUANTITY.finditer(text):
        dimension, factor = UNITS[match.group(2)]
        out.append({"surface": match.group(0), "dimension": dimension,
                    "si": float(match.group(1).replace(",", "")) * factor, "span": match.span()})
    return out


def mask(text, quantities):
    """The question as the reader sees it: each value replaced by a marker, e.g. "A car at <speed1> brakes".

    Speeds are numbered in order, so the reader can say which is the start and which the final speed.
    """
    out, pos, speeds = "", 0, 0
    for q in quantities:
        start, end = q["span"]
        speeds += q["dimension"] == "speed"
        out += text[pos:start] + (f"<speed{speeds}>" if q["dimension"] == "speed" else f"<{q['dimension']}>")
        pos = end
    return out + text[pos:]
