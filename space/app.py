"""Gradio app: ask Newton net a short driving-physics question and get the answer with a trace of how it was worked out.

The pipeline is decisions.md's: extractor → reader + router → assembler → planner → output. The extractor
(model/extract.py), assembler (model/reading.py) and planner (model/planner.py) are the project's own code. The reader
+ router will be the trained TinyLLM; until it is trained, `read` is a keyword-based stand-in with the same interface.

Run from the repo root: uv run --with gradio==6.30.0 python space/app.py
"""
import re
import sys
from pathlib import Path

import gradio as gr

sys.path.append(str(Path(__file__).resolve().parent.parent))  # model/ sits next to app.py on the Space, one up in the repo
from model.extract import extract, mask
from model.planner import solve, trace
from model.reading import DEFAULT_MASS, readings

CODE_URL = "https://github.com/prteek/newton-net"
MAX_CHARS = 160
DIAGRAM = Path(__file__).with_name("architecture.svg").read_text()
REFUSAL = (
    "I can only answer questions about physics of motion, please check:\n"
    "- The query is relevant to the topic\n"
    "- Units are specified correctly e.g. speed (mph, kph, km/h etc)\n"
    "- As much as possible known information is included in the question (e.g. include known speed or acceleration "
    "or distance if not already)"
)
TARGETS = {"v": "speed", "t": "time", "s": "distance", "a": "acceleration", "F": "force"}
NAMES = {"u": "starting speed", "v": "final speed", "a": "acceleration", "t": "time", "s": "distance", "m": "mass",
         "F": "force"}
SI_UNIT = {"u": "m/s", "v": "m/s", "a": "m/s²", "t": "s", "s": "m", "m": "kg", "F": "N"}
SURFACE_DECEL = {"dry": 7.5, "wet": 5.0, "icy": 2.0}  # the generator's typical values; the trained |a| head learns its own

# keyword stand-in for the reader + router: cues looked for in the marked text, lower case
TARGET_CUES = {  # in priority order, looked for in the sentence that asks the question
    "s": r"how far|distance|how much (slip )?road|how much (room|space)|how many (metres|meters|feet|miles|km)",
    "t": r"how long|\btime\b|how many (seconds|minutes|hours)|how soon",
    "v": r"how fast|\bspeed\b|hit (it|them)",
    "F": r"\bforce\b|newtons",
    "a": r"how hard|acceleration|deceleration|\bdecel\b",
}
OFF_TOPIC = (r"\b(fuel|engine|legal|law|tyre|tire|insurance|charg\w*|poem|gear|abs|horsepower|reflex\w*|money|cheap\w*|"
             r"cost|price|safe|safer|mpg|emission\w*)\b|\b(of|for) an? [a-z]+ (golf|gti|tesla|bmw|audi|ford)")
REST = r"from (rest|a standstill|standstill|stationary|a stop)|pull(s|ing)? away|standing start|starting from"
STOP = r"\bstop\w*|stationary|standstill|\bhalt|come to rest|\bahead\b"
BRAKE = r"brak\w*|\bstop\w*|decel\w*|slow\w*|stationary|obstacle|wall|ahead"
ACCEL = r"accelerat\w*|pull(s|ing)? away|floor it|speed up|merg\w*|overtak\w*|thrust|pushed|from (rest|a standstill|stationary)"
CONSTANT = r"constant|cruis\w*|steady|maintain\w*|average speed"
SURFACE = {"icy": r"\bic[ey]\b|icy|snow\w*|frost\w*", "wet": r"rain\w*|\bwet\b|damp"}
REVERSED = r"(to|reach\w*|up to|hit) <speed1>.*from <speed2>"
FINAL = r"(to|reach\w*|up to|hit|get to) <speed1>"


def read(question, quantities):
    """2. Reader + router: what is asked and how the extracted values fit together.

    Returns {"target": planner symbol or "refuse", "reason": why refused, "choice": the reading's choices as in
    model.reading.readings (order, sign, u0, v0, a_source), "a_learned": |a| for a road surface, "surface",
    "probs": each head's probability for its choice}.

    PLACEHOLDER: keyword rules, not the trained TinyLLM. The trained model reads the same marked text and returns the
    same fields, with real probabilities; until then every choice made has probability 1.
    """
    marked = mask(question, quantities).lower()
    text = re.sub(r"<\w+>", "#", marked)  # cues are words, never the value markers
    sentences = [s for s in re.split(r"(?<=[.?!:])\s+", text) if s.strip()]
    asked = next((s for s in sentences if "?" in s), sentences[-1] if sentences else "")
    asked = asked.rsplit(",", 1)[-1] if re.search(r"how|what", asked.rsplit(",", 1)[-1]) else asked
    hits = [t for t, cue in TARGET_CUES.items() if re.search(cue, asked)]
    refuse = None
    if re.search(OFF_TOPIC, text):
        refuse = "off topic"
    elif text.count("?") > 1:
        refuse = "more than one question"
    elif "?" not in text and not re.match(r"(how|what|when)\b", text):
        refuse = "no question asked"
    elif len(hits) > 1 and re.search(r"\band\b", asked):
        refuse = "more than one question"
    elif not hits:
        refuse = "no quantity I can work out is asked for"
    if refuse:
        return {"target": "refuse", "reason": refuse, "choice": None, "a_learned": None, "surface": None,
                "probs": {"target": 1.0}}
    target = hits[0]

    speeds = sum(q["dimension"] == "speed" for q in quantities)
    if speeds == 2:
        order = ("v", "u") if re.search(REVERSED, marked) else ("u", "v")
    elif speeds == 1:
        order = ("v",) if re.search(FINAL, marked) else ("u",)
    else:
        order = ()
    without_rest = re.sub(REST, "", text)
    braking, accelerating = len(re.findall(BRAKE, without_rest)), len(re.findall(ACCEL, text))
    sign = -1 if braking > accelerating else 1
    u0 = "u" not in order and target != "v" and bool(re.search(REST, text))
    v0 = "v" not in order and target != "v" and bool(re.search(STOP, without_rest))
    dims = {q["dimension"] for q in quantities}
    known = set(order) | {s for s, d in (("t", "time"), ("s", "distance")) if d in dims}
    known |= {s for s, implied in (("u", u0), ("v", v0)) if implied}
    surface = next((s for s, cue in SURFACE.items() if re.search(cue, text)), "dry")
    a_source = "none"
    if not {"acceleration", "force"} & dims and target != "a" and len(known & {"u", "v", "t", "s"}) < 3:
        if re.search(CONSTANT, text):
            a_source = "zero"
        elif sign < 0:
            a_source = "learned"
    choice = {"order": order, "sign": sign, "u0": u0, "v0": v0, "a_source": a_source}
    return {"target": target, "reason": None, "choice": choice, "a_learned": SURFACE_DECEL[surface],
            "surface": surface, "probs": {"target": 1.0, "order": 1.0, "sign": 1.0, "implied": 1.0}}


def assemble(quantities, reading):
    """3. Assembler: the known values for the reader's choices, or (None, (stage, reason))."""
    choice = reading["choice"]
    options, refusal = readings(quantities, reading["target"], reading["a_learned"], sign=choice["sign"])
    if refusal:
        return None, refusal
    keys = ("order", "u0", "v0", "a_source")
    known = next((k for c, k in options if all(c[x] == choice[x] for x in keys)), None)
    return (known, None) if known is not None else (None, ("reader", "the reading chosen isn't possible"))


def number(x):
    return f"{x:,.0f}" if abs(x) >= 100 else f"{x:.3g}"


def display(symbol, x):
    """Answer in display units: s or h; m or km; km/h; m/s²; N."""
    if symbol == "t":
        return f"{number(x / 3600)} h" if x > 3600 else f"{number(x)} s"
    if symbol == "s":
        return f"{number(x / 1000)} km" if x > 2000 else f"{number(x)} m"
    if symbol in ("u", "v"):
        return f"{number(x * 3.6)} km/h"
    return f"{number(x).replace('-', '−')} {SI_UNIT[symbol]}"


def friendly(reason):
    """A planner refusal in words: what's missing, what contradicts, or why the motion can't happen."""
    if reason.startswith("missing inputs: "):
        options = re.findall(r"\{([^}]*)\}", reason)
        sets = [" and ".join(NAMES[s] for s in o.split(", ")) for o in options]
        return "Not enough information. Add the " + ", or the ".join(sets) + "."
    if reason.startswith("conflicting inputs: "):
        return "The values given contradict each other: " + reason.removeprefix("conflicting inputs: ") + " (SI)."
    return "That motion isn't possible: " + reason + "."


def assumptions(reading, known, quantities, used):
    c, out = reading["choice"], []
    if c["u0"] and "u" in used:
        out.append("starting from rest → u = 0 m/s")
    if c["v0"] and "v" in used:
        out.append("comes to a stop → v = 0 m/s")
    if c["a_source"] == "zero" and "a" in used:
        out.append("constant speed → a = 0 m/s²")
    if c["a_source"] == "learned" and "a" in used:
        out.append(f"no deceleration given, {reading['surface']} road → a = −{reading['a_learned']} m/s² "
                   "(typical value)")
    if "m" in used and not any(q["dimension"] == "mass" for q in quantities):
        out.append(f"mass not given → {DEFAULT_MASS:,.0f} kg")
    return out


def run(question):
    """The whole pipeline: a dict with the stage that refused and why, or the answer and its trace."""
    question = question or ""
    quantities = extract(question)
    out = {"quantities": quantities, "marked": mask(question, quantities)}
    if problems := [q for q in quantities if q["problem"]]:
        return out | {"refused_by": "extractor", "reason": "; ".join(f"{q['surface']}: {q['problem']}" for q in problems)}
    reading = read(question, quantities)
    out["reading"] = reading
    if reading["target"] == "refuse":
        return out | {"refused_by": "reader", "reason": reading["reason"]}
    known, refusal = assemble(quantities, reading)
    if refusal:
        return out | {"refused_by": refusal[0], "reason": refusal[1]}
    out["known"] = known
    target = reading["target"]
    steps, values, reason = solve(known, target)
    if steps is None:
        return out | {"refused_by": "planner", "reason": reason}
    used = {i for _, ins, *_ in steps for i in ins} | ({target} if not steps else set())
    return out | {"refused_by": None, "answer": values[target], "steps": trace(steps, values),
                  "assumptions": assumptions(reading, known, quantities, used)}


def values_table(r):
    rows = ["| as written | quantity | SI |", "|---|---|---:|"]
    for q in r["quantities"]:
        rows.append(f"| {q['surface']} | {q['dimension'] or '?'} | " + (f"{q['si']:.6g}" if q["si"] is not None else "–")
                    + " |")
    return "\n".join(rows)


def reader_lines(r):
    reading = r["reading"]
    if reading["target"] == "refuse":
        return [f"- target: refuse ({reading['reason']}), probability {reading['probs']['target']:.2f}"]
    c = reading["choice"]
    speeds = [q for q in r["quantities"] if q["dimension"] == "speed"]
    lines = [f"- target: {TARGETS[reading['target']]} ({reading['target']}), "
             f"probability {reading['probs']['target']:.2f}"]
    lines += [f"- {q['surface']} is the {NAMES[role]} ({role})" for q, role in zip(speeds, c["order"])]
    if c["a_source"] == "zero":
        lines.append("- motion: constant speed")
    elif c["a_source"] == "learned" or any(q["dimension"] in ("acceleration", "force") for q in r["quantities"]):
        lines.append(f"- motion: {'braking (−)' if c['sign'] < 0 else 'accelerating (+)'}")
    return lines


def respond(question):
    """(answer markdown, trace markdown) for the app."""
    r = run(question)
    if r["refused_by"] == "planner":
        head = f"**Can't answer this one.** {friendly(r['reason'])}"
    elif r["refused_by"]:
        head = REFUSAL + f"\n\n*Refused by the {r['refused_by']}: {r['reason']}.*"
    else:
        target = r["reading"]["target"]
        motion = " (braking)" if target in ("a", "F") and r["answer"] < 0 else ""
        head = f"## {TARGETS[target].capitalize()}: {target} = {display(target, r['answer'])}{motion}"
    parts = [f"**As the encoder sees it:** {r['marked']}", "**Values read** (extractor)", values_table(r)]
    if "reading" in r:
        parts += ["**Reader + router** (keyword stand-in: confidence is 1 until the model is trained)",
                  "\n".join(reader_lines(r))]
    if "known" in r:
        parts += ["**Known values** (assembler, SI): " + ", ".join(f"{k} = {v:.6g} {SI_UNIT[k]}"
                                                                   for k, v in r["known"].items())]
    if r["refused_by"]:
        parts.append(f"**Refused by the {r['refused_by']}:** {r['reason']}")
    else:
        target = r["reading"]["target"]
        if r["assumptions"]:
            parts += ["**Assumptions**", "\n".join(f"- {a}" for a in r["assumptions"])]
        steps = "\n".join(f"{i}. `{s}`" for i, s in enumerate(r["steps"], 1)) or f"{target} is given"
        parts += ["**Planner steps** (SI)", steps,
                  f"**Answer in SI:** {target} = {r['answer']:.6g} {SI_UNIT[target]}"]
    return head, "\n\n".join(parts)


EXAMPLES = [
    "I'm doing 70 mph and I want to go to 10 mph in 5 s. How much force is needed to stop?",
    "A car at 30 m/s brakes at 0.7g. How long until it stops?",
    "A car brakes at 6 m/s². How far does it travel before stopping?",
    "A 1,500 kg car goes from 30 mph to 60 mph in 6 s. What net force acts on it?",
    "cruising at 70 mph for 20 minutes, how far will I get?",
    "there's a dog 30 m ahead and I'm at 40 mph. how hard do I need to brake?",
    "How long does it take to reach 60 mph?",
    "A car at 20 m/s brakes at 5 m/s² for 6 s. How far does it travel?",
    "doing 70 in the rain, how far to stop?",
    "Is it legal to drive at 80 mph on a motorway?",
]

with gr.Blocks(title="Newton net") as demo:
    gr.HTML(
        '<div style="padding:10px 14px;border-radius:8px;border:1px solid #d97706;background:rgba(245,158,11,0.15)">'
        "<b>Demo only:</b> this is not the real model. Questions are read by keyword rules standing in for the "
        "TinyLLM, which hasn't been trained yet; the extractor and planner are the project's real code.</div>"
    )
    gr.Markdown(
        "# Newton net: exact driving physics from a question\n"
        "Ask one short question about braking, accelerating or cruising. A reader works out what is asked and what is "
        "known, and a fixed planner chains the equations of motion to an exact answer, with a trace of every step or a "
        "refusal with the reason."
    )
    gr.HTML(DIAGRAM)
    gr.Markdown(
        "*The reader is a keyword placeholder until the TinyLLM is trained. Assumes dry/wet/icy braking at "
        "7.5/5.0/2.0 m/s², no reaction time, and a 2,000 kg car unless stated.*"
    )
    question = gr.Textbox(label=f"Your question (up to {MAX_CHARS} characters)", max_length=MAX_CHARS, lines=2,
                          placeholder="doing 70 mph in the rain, how far to stop?")
    gr.Examples(EXAMPLES, question)
    ask = gr.Button("Answer", variant="primary")
    answer = gr.Markdown()
    with gr.Accordion("Trace", open=True):
        steps = gr.Markdown()
    gr.Markdown(f"Code: [{CODE_URL.removeprefix('https://')}]({CODE_URL}).")
    gr.on([ask.click, question.submit], respond, question, [answer, steps])

if __name__ == "__main__":
    demo.launch()
