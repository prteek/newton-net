"""3. Assembler: the extracted values and the reader's choices -> one known value per quantity, in SI.

    assemble([{"dimension": "speed", "si": 31.3}, {"dimension": "acceleration", "si": 6.9}],
             {"order": ("u",), "sign": -1, "u0": False, "v0": True, "a_source": None})
    -> {"m": 2000.0, "u": 31.3, "a": -6.9, "v": 0.0}

Returns None when the values don't fit the choices (e.g. two values for one quantity).
"""

DEFAULT_MASS = 2000.0  # kg, when the question doesn't say
SYMBOL = {"acceleration": "a", "time": "t", "distance": "s", "mass": "m", "force": "F"}


def assemble(quantities, choice):
    speeds = [q["si"] for q in quantities if q["dimension"] == "speed"]
    others = [(SYMBOL[q["dimension"]], q["si"]) for q in quantities if q["dimension"] != "speed"]
    if len(speeds) != len(choice["order"]) or len({s for s, _ in others}) < len(others):
        return None
    known = {"m": DEFAULT_MASS} | dict(zip(choice["order"], speeds)) | dict(others)
    for q in ("a", "F"):  # stated as a size; the reader says which way
        if q in known:
            known[q] = choice["sign"] * abs(known[q])
    if choice["u0"]:
        known["u"] = 0.0
    if choice["v0"]:
        known["v"] = 0.0
    if choice["a_source"] is not None:
        known["a"] = choice["sign"] * choice["a_source"]
    return known
