---
title: Project decisions
updated: 2026-10-29
---

Recorded from the 2026-10-29 design chat (`history.md`, sections 5–8).

# Scope

- Routes name the quantity asked for, not a formula: speed, time, distance, acceleration, force, or refusal (see System requirements → 2. Reader + router). The planner chooses the formulas. Without a refusal route the model would make up answers for off-topic questions.
- Motion is not just braking: each question is one of three motion types, braking, accelerating or constant speed, and every route covers each type where it makes sense (e.g. distance to stop, distance to reach 60 mph from rest, distance covered in 20 minutes at 70 mph). Acceleration is signed in the engine: negative when braking.
- Reaction time is out of scope: there is no reaction phase, so braking starts immediately and time is a single variable t. Questions that mention a reaction time are not generated.
- Inputs are ≤160-character, Twitter-like questions.
- This version is a proof of concept, not a benchmark: no comparison against baselines (tool-calling LLMs, separately trained classifier + engine). The equations are simple enough that any LLM or large enough network could learn them, so beating one here would prove little. The aim is to show the architecture works and can extend to more complex scenarios. Real benchmarking happens on those complex scenarios, and should weigh the simplicity and robustness this design offers over the alternatives, not just accuracy.
- Decision support only, not a controller: a verified deterministic controller must decide and act.

# Architecture

- Physics runs inside `TinyLLM.forward()` with gradients flowing through it. This is what makes it not tool calling and lets the reading of values be learned from answers; the choice of target is learned from its label (see System requirements → 2. Reader + router).
- TinyLLM: d_model 48, 1–2 layers, 48–64 tokens, 2k–8k vocab (~0.25–0.5M params).
- The model extracts value + unit; a deterministic normaliser converts to SI. Unit conversion is never learned.
- Acceleration magnitude (braking or speeding up) should be continuous and not discrete.
- We will focus on different mean levels due to unknown/uncertain inputs but not care about uncertainty or propagating it in the first version
- Action head (answer / refuse) we will either answer or reject a request even in cases of ambiguity

# Training

- Train jointly, this is in contrast to training a claseeifier and modelling physics separately and gluing them together
- Joint training lets improvement of both parts together and allows the parser (TinyLLM) to learn friction variation from words like "wet road" when not explicitly given among other things
- Start with full supervision (route, slots, units, action, target); answer-only supervision risks shortcuts
- The answer loss is on log|answer| (e.g. MSE between log|engine output| and log|target|), not raw SI. Answers span 1 m/s² to 86 km once constant-speed journeys are in, so a raw-SI loss would be dominated by long journeys; the log makes errors relative (0.1 in log ≈ 10% off) at every scale. The engine still outputs SI; only the loss is logged. Sign is not part of the log: it follows from the motion type (negative acceleration when braking).

# Deployment

- TBD either within app itself or endpoint depending on model size

# User requirements

## Input

- They type one question of up to 160 characters (hard limit), in free text (english) with any/no units: "doing 70 mph in
    the rain, how far to stop?"

## Output

- The answer with unit relevant to question (unit should be output as part of the equation itself) e.g. "98 m"

- A generic refusal stating most common (hard coded) failure modes
    I can only answer questions about physics of motion please check:
    - The query is relevant to the topic
    - Units are specified correctly e.g. speed (mph, kph, km/h etc)
    - As much as possible known information is included in the question (e.g. include known speed or acceleration or distance if not already)
- A trace, which is what makes this different from asking an LLM:
     - Internal answer with SI units
     - Route i.e. equation used: "distance / acceleration / time".
     - Optional:
        - Assumptions: "wet road → deceleration 5.0 m/s²", "Mass -> 2000kg"
        - Confidence: how sure model was ? the route probabilities, e.g. "stopping distance 0.97".
- A static architecture diagram of the model at the top with some writeup on what the app does

# System requirements

The system is a pipeline of six stages. Only the reader + router is learned; everything else is fixed code. Each stage has one job, and any stage can refuse, with a reason that travels to the output.

## Pipeline

```
Question (≤160 chars)
  │
  ▼
1. Checker ──────────── too long, not English ───────────────────────────► Refusal
  │
  ▼
2. Reader + router: TinyLLM (learned)
  ├─ target head:    speed | time | distance | acceleration | force | refuse ──► Refusal
  ├─ slot tagger:    value + unit for u, v, a, t, s, m, F, with their spans
  └─ implied values: stop → v = 0, from rest → u = 0, cruising → a = 0,
                     brake → a < 0, road surface → |a|, no mass → 2,000 kg
  │
  ▼
3. Assembler ────────── two values for one quantity ─────────────────────► Refusal
  │                     (stated values win over implied ones)
  ▼
4. Unit converter ───── unit not in the list ─────────────────────────────► Refusal
  │                     value + unit → SI
  ▼
5. Planner: model/planner.py (fixed, differentiable)
  ├─ conflict check ─ inputs contradict each other ──────────────────────► Refusal
  └─ shortest chain of guarded rules ─ missing input, guard fails ───────► Refusal
  │
  ▼
6. Output formatter: answer in display units, trace, assumptions made
```

Training signals:
- Target head: cross-entropy on the labelled target.
- Slot tagger and implied values: supervised on their labels, and also by the answer loss on log|answer|, which flows back through the planner's steps to the values they used.

## Why both a router and a planner
Each does the half of the job the other can't.
- **Why a router:** turning free text into structure needs judgement. "Doing 70 in the rain" means u = 70 mph and a wet road; "how hard do I need to brake" asks for acceleration; "what's the 0–60 time of a Golf" is off topic despite the numbers. Rules break on this kind of variety, so it is learned.
- **Why a planner:** once the question is structured, the answer is fixed by physics. Learning it would make answers approximate, need data for every combination of inputs and targets, and give no reason when something can't be computed. The planner is exact, needs no data, finds a chain for any combination the equations allow, and explains its refusals.
- **How they fit together:**
  - **Division of labour:** the router decides *what is asked and what is known*; the planner decides *how to get from one to the other*. The router never picks a formula, and the planner never reads text.
  - **Refusals split by who can judge them:** the router refuses on meaning (off topic, ambiguous, two questions), the planner on maths (missing input, contradiction, impossible motion), each with its own reason.
  - **The planner trains the router:** it is differentiable, so the answer loss flows back through the planned steps to the values the router read. A misread speed shows up as a wrong answer, and the gradient points at the speed.
  - **Growth goes to the planner:** a new equation (friction, power) is one rule, and every combination it allows works without new routes. The router only needs new data when there is a new kind of quantity to read or a new target to name.
  - **One trace from both:** the router's part (what it read, what it assumed, how sure it was) and the planner's part (each step with its numbers) together make the trace the user sees.

## 1. Checker
- The text box accepts at most 160 characters; longer input is refused before the model runs.
- Input with non-English characters is refused.

## 2. Reader + router (learned)
Gives the rest of the system three things; anything else is the planner's job.
- **Target:** probabilities over speed, time, distance, acceleration, force and refuse. Probabilistic and learned, but trained on the target label (cross-entropy), not through the answer: answers for different targets have different units and can't be blended, so the answer loss can't say which target was right. At inference the most likely target is used, and its probability is shown as confidence.
- **Refusals on meaning** (the refuse class): off topic, very ambiguous, more than one question, no question asked.
- **Stated values:** for each number in the question, which quantity it is (u, v, a, t, s, m, F) and its unit, with character spans for the trace. Whether the number is an acceleration or a deceleration sets the sign of a (negative when braking).
- **Implied values,** as numbers the planner can use, each reported as an assumption in the trace:

  | Not stated | Value used |
  |---|---|
  | Starting speed, with "from rest", "stationary" etc. | u = 0 |
  | Final speed, when the question is about stopping | v = 0 |
  | Acceleration, at constant speed ("cruising", "steady") | a = 0 |
  | Acceleration, braking with a road surface given | learned value for that surface |
  | Acceleration, braking with no surface given | learned value for a dry road |
  | (both acceleration rows apply only when nothing else fixes a: not when a is what's asked, nor when the question says where the car must stop) | |
  | Mass | 2,000 kg, unless stated |
  | Anything else (speed when not from rest, distance or time) | nothing: the planner refuses with what's missing |

- **Not its job:** choosing a formula, labelling the motion type (it survives only as implied values and the sign of a), or refusing for missing inputs or contradictions.

## 3. Assembler
- Combines stated and implied values into one value per quantity. A stated value replaces an implied one.
- Two stated values for the same quantity ("stationary at 20 m/s") are refused: the planner only ever sees one value per quantity, so it can't catch this.

## 4. Unit converter
- Converts each value + unit to SI with fixed factors; conversion is never learned.
- A unit not in the list is refused. The list isn't fixed yet; the data generator currently uses m/s, mps, km/h, kph, mph (speed); m/s², g (acceleration); s, seconds, min, minutes, hours (time); m, metres, ft, km, miles (distance); kg, tonnes (mass); N, kN (force).

## 5. Planner
Implemented in `model/planner.py`.
- **Takes:** a target name and a dict of known values in SI, keyed u, v, a, t, s, m, F, with a signed and implied values already filled in. Values can be floats or torch tensors, so gradients flow through a plan.
- **Returns:** the steps applied (each a rule with its formula text) and every value computed, or a refusal reason.
- **Rules:** each equation solved for each unknown:
  - speed: v = u + a·t, v = √(u² + 2as), v = 2s/t − u
  - acceleration: a = F/m, a = (v − u)/t, a = (v² − u²)/2s, a = 2(s − u·t)/t²
  - distance: s = u·t + a·t²/2, s = (v² − u²)/2a, s = (u + v)/2 · t
  - time: t = (v − u)/a, t = 2s/(u + v), t = 2s/(u + √(u² + 2as)) (which also covers a = 0: t = s/u)
  - force: F = m·a

  The three in v = √(u² + 2as), v = 2s/t − u and s = (u + v)/2 · t were added so that any three of u, v, a, t, s give the other two.
- **Search:** breadth-first, so the plan is the shortest chain of rules from the known values to the target; rules are tried in a fixed order, so the same question always gets the same plan.
- **Extra inputs are checked, not ignored:** before planning, everything the inputs determine is derived, and every rule that can recompute a value from the others is compared with it. A gap of more than 5% (or 0.05 in SI near zero) is refused as a conflict naming the value and the rule, e.g. "v is given as 26.82, but v = u + a·t gives 12".
- **Guards:** each rule checks its inputs before it runs: t, s and m positive; a ≠ 0 and u + v ≠ 0 where they divide; the car doesn't stop before covering s (u² + 2as ≥ 0); a braking car doesn't run past the moment it stops. Every result is checked: t > 0, s ≥ 0, v ≥ 0. A guarded rule is skipped so another can still reach the target; if none can, the guard's reason is the refusal.
- **Tolerances:** |x| < 0.001 (SI) counts as zero, since values read by a trained model are rarely exactly 0. Stopping within 5% of the distance or time asked about counts as reaching it (at v = 0), since stated values are rounded.
- **Missing inputs** are named as inputs the user could add, not intermediate results, e.g. "{a} or {s, u} or {t, u}".
- **Not its job:** reading text, converting units, filling in defaults, or judging whether a question is on topic.

## 6. Output formatter
- **Answer** in display units, as part of the final equation (e.g. "98 m"):

  | Quantity | Unit |
  |---|---|
  | Time | s; hours above 3,600 s |
  | Distance | m; km above 2,000 m |
  | Speed | km/h always |
  | Acceleration | m/s² |
  | Force | N |

- **No plausibility check:** any answer the planner produces is shown, as this is a proof of concept.
- **Trace:** answer in SI, target and its probability, values read with their units, assumptions made, and each planned step with its numbers.
- **Refusal:** the fixed message from User requirements → Output, plus the specific reason from the stage that refused.

## Refusals at a glance

| Reason | Stage |
|---|---|
| Longer than 160 characters; not English | 1. Checker |
| Off topic, very ambiguous, more than one question, no question asked | 2. Reader + router |
| Two values for one quantity | 3. Assembler |
| Unit not in the list | 4. Unit converter |
| Inputs contradict each other; missing input; impossible motion (guard) | 5. Planner |
