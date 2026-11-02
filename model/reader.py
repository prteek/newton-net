"""2. Reader + router: what is asked, and how the extracted values fit together.

PLACEHOLDER: keyword rules standing in for the TinyLLM, which will read the same marked text and return the same
choices once it is trained.

    read("A car at <speed1> brakes at <acceleration>. How long until it stops?")
    -> {"target": "t", "order": ("u",), "sign": -1, "u0": False, "v0": True, "a_source": None}

Returns None when the question isn't one it can answer.
"""

import re

# what is asked, by priority: the first cue found picks the target
TARGET_CUES = {
    "s": r"how far|distance|how much (slip )?road",
    "t": r"how long|\btime\b",
    "v": r"how fast|\bspeed\b",
    "F": r"\bforce\b",
    "a": r"how hard|acceleration|deceleration",
}
OFF_TOPIC = r"\b(fuel|engine|legal|tyres?|insurance|poem|gear)\b"
REST = r"from (rest|a standstill|stationary)|pulls? away"
STOP = r"\bstop|stationary|\bahead\b"  # "a dog 30 m ahead": stop before it
BRAKE = r"\bbrak|\bstop|decel"
CONSTANT = r"constant|cruis|steady"
FINAL = r"(\bto|reach\w*) <speed1>"  # a lone speed that is where the car ends up
SURFACE_DECEL = {"icy": 2.0, "wet": 5.0, "dry": 7.5}  # typical braking on each road, m/s²
SURFACE_CUES = {"icy": r"\bic[ey]|snow", "wet": r"rain|\bwet\b"}


def read(marked):
    text = marked.lower()
    words = re.sub(r"<\w+>", "#", text)  # cues are words, never the value markers ("<time>" isn't "time")
    target = next((t for t, cue in TARGET_CUES.items() if re.search(cue, words)), None)
    if target is None or re.search(OFF_TOPIC, words):
        return None
    speeds = text.count("<speed")
    order = {0: (), 1: ("v",) if re.search(FINAL, text) else ("u",), 2: ("u", "v")}.get(speeds)
    if order is None:
        return None
    braking = bool(re.search(BRAKE, text))
    v0 = target != "v" and "v" not in order and bool(re.search(STOP, text))
    u0 = "u" not in order and bool(re.search(REST, text))
    # with no acceleration or force given: cruising means a = 0, braking means the road's typical deceleration,
    # unless the speeds, time and distance already fix it
    a_source = None
    kinematics = len(order) + u0 + v0 + ("<time>" in text) + ("<distance>" in text)
    if "<acceleration>" not in text and "<force>" not in text and target != "a" and kinematics < 3:
        if re.search(CONSTANT, text):
            a_source = 0.0
        elif braking:
            surface = next((s for s, cue in SURFACE_CUES.items() if re.search(cue, text)), "dry")
            a_source = SURFACE_DECEL[surface]
    return {"target": target, "order": order, "sign": -1 if braking else 1, "u0": u0, "v0": v0, "a_source": a_source}
