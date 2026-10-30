---
title: Project decisions
updated: 2026-10-29
---

Recorded from the 2026-10-29 design chat (`history.md`, sections 5–8).

# Scope

- Routes name the quantity asked for, not a formula: speed, time, distance, acceleration, force, or refusal (see System requirements → 3. Reader + router). The planner chooses the formulas. Without a refusal route the model would make up answers for off-topic questions.
- Motion is not just braking: each question is one of three motion types, braking, accelerating or constant speed, and every route covers each type where it makes sense (e.g. distance to stop, distance to reach 60 mph from rest, distance covered in 20 minutes at 70 mph). Acceleration is signed in the engine: negative when braking.
- Reaction time is out of scope: there is no reaction phase, so braking starts immediately and time is a single variable t. Questions that mention a reaction time are not generated.
- Inputs are ≤160-character, Twitter-like questions.
- This version is a proof of concept, not a benchmark: no comparison against baselines (tool-calling LLMs, separately trained classifier + engine). The equations are simple enough that any LLM or large enough network could learn them, so beating one here would prove little. The aim is to show the architecture works and can extend to more complex scenarios. Real benchmarking happens on those complex scenarios, and should weigh the simplicity and robustness this design offers over the alternatives, not just accuracy.
- Decision support only, not a controller: a verified deterministic controller must decide and act.

# Architecture

- Physics runs inside `TinyLLM.forward()` with gradients flowing through it. This is what makes it not tool calling and lets the model learn how to read a question from its answer alone (see System requirements → 3. Reader + router).
- TinyLLM: d_model 48, 1–2 layers, 48–64 tokens, 2k–8k vocab (~0.25–0.5M params).
- A fixed extractor (quantulum3 + pint) finds each value and unit and converts it to SI; the unit's dimension fixes which quantity a value can be. Extraction and unit conversion are never learned.
- Acceleration magnitude (braking or speeding up) should be continuous and not discrete.
- We will focus on different mean levels due to unknown/uncertain inputs but not care about uncertainty or propagating it in the first version
- Action head (answer / refuse) we will either answer or reject a request even in cases of ambiguity

# Training

- Train jointly, this is in contrast to training a claseeifier and modelling physics separately and gluing them together
- Joint training lets improvement of both parts together and allows the parser (TinyLLM) to learn friction variation from words like "wet road" when not explicitly given among other things
- Training data is only the question, its answer and the answer's unit (null for questions the system shouldn't answer at all). Nothing else is labelled: not the target, which number is which quantity, the sign or implied values. Keeps data simple, and lets paraphrased or real question–answer pairs be used as they are.
- The target comes from the answer's unit (m → distance, m/s → speed, …; no unit → refuse) and is trained by cross-entropy. Everything else the reader + router decides is learned by matching the answer through the planner (see System requirements → Training).
- The answer loss is on log|answer| (e.g. MSE between log|engine output| and log|target|), not raw SI. Answers span 1 m/s² to 86 km once constant-speed journeys are in, so a raw-SI loss would be dominated by long journeys; the log makes errors relative (0.1 in log ≈ 10% off) at every scale. The engine still outputs SI; only the loss is logged. The sign is compared separately: a braking acceleration or force is negative.

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
2. Extractor: model/extract.py (fixed: quantulum3 + pint)
  │                     each number + unit → dimension and SI value;
  │                     the text with each value replaced by a marker: "at <speed1> brakes at <acceleration>"
  ├──────────────────── number without a unit, unit of another dimension ─► Refusal
  ▼
3. Reader + router: TinyLLM (learned), reads the marked text
  ├─ target head:  speed | time | distance | acceleration | force | refuse ──► Refusal
  ├─ order head:   which speed is the start (u) and which the final speed (v)
  ├─ sign head:    braking (−) | accelerating (+); soft (2p − 1) in training
  └─ implied:      from rest → u = 0, stop → v = 0,
                   no a stated → none | 0 (cruising) | road surface → |a| (learned value)
  │
  ▼
4. Assembler: model/reading.py (fixed)
  │                     one value per dimension (up to two speeds); a and F → sign × |value|;
  │                     implied values fill gaps; mass 2,000 kg unless stated
  ├──────────────────── two values of one dimension, three speeds ────────► Refusal
  ▼
5. Planner: model/planner.py (fixed, differentiable)
  ├─ conflict check ─ inputs contradict each other ──────────────────────► Refusal
  └─ shortest chain of guarded rules ─ missing input, guard fails ───────► Refusal
  │
  ▼
6. Output formatter: answer in display units, trace, assumptions made
```

## Training
Training data is only (question, answer, unit); see `dataset/generate.py`.
- **Target head:** cross-entropy on the target named by the answer's unit (no unit → refuse).
- **Order and implied heads:** by matching the answer. Together with the sign, their choices make a *reading*: one complete set of known values. A question has at most 24 readings (2 speed orders × u = 0 or not × v = 0 or not × 3 ways to fill a), 5.7 on average. The planner runs on each; a reading *matches* if it reproduces the labelled answer (within 5% on log|answer|, same sign), or refuses when the label has no answer. The loss raises the probability of the matching readings: −log Σ P(reading) · match(reading), where P(reading) is the product of the heads' probabilities.
- **Why match answers instead of labelling each choice:**
  - **Simple data.** Each question needs only its answer and unit. Labelling which number is which quantity, the sign and every implied value is the expensive, error-prone part of making data, and it doesn't survive paraphrasing: an LLM rewrite or a real question–answer pair comes with an answer, not with labels.
  - **The extractor makes the search small.** The unit fixes every value's quantity except which speed is the start and which the end, so only a handful of readings exist per question, few enough to run the planner on all of them.
  - **The physics does the judging.** A wrong reading usually fails outright (a negative distance, a time before the start, a conflict) rather than giving a near miss, so the planner is a strong teacher. This is what makes joint training real: the router learns to read questions from whether the physics gets the answer right.
  - **Cost:** about 6 planner runs per question in training, noisier learning than with direct labels, and choices the answer can't tell apart (below).
- **Sign head: soft sign, trained through the physics.** In training the sign is not enumerated: the sign head's expected sign 2p − 1 (p = P(accelerating)) multiplies a stated a and F, and a road-surface |a|, as a number. So the sign head gets gradient through the planner's equations, like |a|: braking at 5 m/s² read with p = 0.1 gives a = −4 m/s² and a stopping time of 5 s instead of 4 s, and the gradient pushes p down. At inference the sign is hard (±1), so stated values are used exactly. It halves the readings per question.
  - **Catch: a wrong sign can't be fixed by gradient alone.** Stopping distance (−u²/2a) and stopping time (−u/a) depend on 1/a, so as 2p − 1 moves through 0 on the way from the wrong sign to the right one, the answer goes to infinity, and the planner's guards refuse readings with the wrong sign (a negative distance), which then give no gradient.
  - **Warm-up:** start training with the sign enumerated as −1 and +1 inside the readings (`readings(..., sign=None)`) so the sign head learns which side it is on from matching answers, then switch to the soft sign once it is confident, to fine-tune it through the physics.
- **Road-surface |a|:** a continuous value, trained by the answer loss through the planner's arithmetic.
- **What the answer can't tell apart:** with the sign enumerated, more than one reading matches in 39% of generated questions. Mostly time symmetry: braking from 124 km/h to a stop covers the same distance and takes the same time as accelerating from rest to 124 km/h, so for distance and time questions the answer can't say which. The answer is right either way, but the trace can show the mirror reading. Signed answers (acceleration, force) do tell them apart and teach the shared encoder what "brake" means.
- **Refused questions:** off-topic and other reader refusals train only the target head (refuse); questions refused later (missing input, contradiction, impossible motion) train the target head on their unit and push the readings towards ones the planner refuses.

## Why both a router and a planner
Each does the half of the job the other can't.
- **Why a router:** turning free text into structure needs judgement. "Doing 70 in the rain" means u = 70 mph and a wet road; "how hard do I need to brake" asks for acceleration; "what's the 0–60 time of a Golf" is off topic despite the numbers. Rules break on this kind of variety, so it is learned.
- **Why a planner:** once the question is structured, the answer is fixed by physics. Learning it would make answers approximate, need data for every combination of inputs and targets, and give no reason when something can't be computed. The planner is exact, needs no data, finds a chain for any combination the equations allow, and explains its refusals.
- **How they fit together:**
  - **Division of labour:** the router decides *what is asked and what is known*; the planner decides *how to get from one to the other*. The router never picks a formula, and the planner never reads text.
  - **Refusals split by who can judge them:** the router refuses on meaning (off topic, ambiguous, two questions), the planner on maths (missing input, contradiction, impossible motion), each with its own reason.
  - **The planner trains the router:** the router's choices are scored by whether the planner turns them into the right answer, and the road-surface |a| is trained through the planner's arithmetic. A misread speed order or sign shows up as a wrong answer or a refusal, and the loss moves probability away from it.
  - **Growth goes to the planner:** a new equation (friction, power) is one rule, and every combination it allows works without new routes. The router only needs new data when there is a new kind of quantity to read or a new target to name.
  - **One trace from both:** the extractor's values, the router's choices (which speed is which, the sign, what it assumed, how sure it was) and the planner's steps together make the trace the user sees.

## 1. Checker
- The text box accepts at most 160 characters; longer input is refused before the model runs.
- Input with non-English characters is refused.

## 2. Extractor
Implemented in `model/extract.py`; nothing is learned.
- quantulum3 finds each number + unit (including number words such as "seventy"); pint converts it to SI and gives its dimension: speed, acceleration, time, distance, mass or force. Any unit pint knows is accepted, so knots, ft/s or yards need no list.
- **Fixes for quantulum3:** "g" after a number is acceleration (standard gravity), not grams; "seconds" and "minutes" are times, not angles; "to" or a dash between two values ("15 m/s to 45 mph", "0–60 mph") is split into both values, with a bare first number taking the second's unit; a unit that swallowed the next word ("27 km/h at" read as per ampere-turn) is parsed again without it.
- **Refuses:** a number with no unit ("doing 70 in the rain") and a unit of a dimension the physics doesn't cover ("150 hp", "a 5 degree slope").
- **Not its job:** deciding which speed is the start or final speed, or the sign. The dimension fixes everything else: a time can only be t, a distance only s, a mass only m, a force only F.
- **Marks the text for the encoder** (`mask`, in the first version): before tokenising, each value is replaced by a marker of its dimension, and the encoder never sees the numbers or units:

  "A 1,336 kg car goes from 15 m/s to 45 mps in 7 s. What net force acts on it?"
  → "A <mass> car goes from <speed1> to <speed2> in <time>. What net force acts on it?"

  Speeds are numbered in order, because the order head reads the vectors at `<speed1>` and `<speed2>`; a value the extractor can't use becomes `<number>`. Rationale:
  - **No double work.** The extractor has already read the number and unit exactly. Making the encoder read "1,336", "mps" or "0.69g" again only adds things for it to get wrong.
  - **The encoder learns language, not numerals.** What it has to decide (what is asked, which speed is the start, braking or accelerating, what is implied) depends on the words around the values, not their digits. "From 15 m/s to 45 mps" and "from 30 mph to 70 mph" become the same input, so one example teaches both, and the model can't latch onto particular numbers or unit spellings in the generated data.
  - **A smaller, closed vocabulary.** Numbers and unit spellings are open-ended (every value is a new token); a handful of markers replace them, which suits a 2k–8k vocabulary and a tiny model.
  - **A place for the order head to look.** `<speed1>` and `<speed2>` are fixed positions whose vectors say how each speed is used ("from …", "reach …", "doing …").
  - **The values stay exact.** They go from the extractor straight to the assembler and planner, never through the network, so masking costs no precision.
  - **Cost:** the encoder can't use a value's size as a clue (e.g. that 0 in "0 to 60 mph" means from rest; the 0 still reaches the planner as a speed). Within the questions in scope, the words carry the meaning.

## 3. Reader + router (learned)
Decides only what the extractor leaves open; anything else is the planner's job.
- **Target:** probabilities over speed, time, distance, acceleration, force and refuse; trained on the target the answer's unit names. At inference the most likely target is used, and its probability is shown as confidence.
- **Refusals on meaning** (the refuse class): off topic, very ambiguous, more than one question, no question asked.
- **Order:** for two speeds, which is the start (u) and which the final speed (v); for one speed, which of the two it is.
- **Sign:** braking (−) or accelerating (+), one per question, applied to a stated a and F. Soft in training (the expected sign 2p − 1), hard at inference (see Training). Words such as "brakes", "decel", "stop within" point to braking; "accelerates", "floor it", "pulls away" to accelerating. When a or F is the target, the planner's physics gives its sign instead.
- **Implied values,** each reported as an assumption in the trace:

  | Not stated | Choice |
  |---|---|
  | Starting speed, with "from rest", "stationary" etc. | u = 0 |
  | Final speed, when the question is about stopping | v = 0 |
  | Acceleration, at constant speed ("cruising", "steady") | a = 0 |
  | Acceleration, braking on a road surface (or none given) | sign × learned \|a\| for that surface |
  | Anything else (speed when not from rest, distance or time) | nothing: the planner refuses with what's missing |

- **Trained by:** the target on its label; order and implied values by matching the answer; the sign and the road-surface |a| through the planner's equations (see Training). None of them has its own label.

## 4. Assembler
Implemented in `model/reading.py`, which also lists every reading for training.
- Combines the extractor's values with the router's choices into one value per quantity, in SI.
- Applies the sign to a stated a and F: sign × |value|, so a value written with a minus isn't flipped twice. The sign is the expected sign 2p − 1 in training and ±1 at inference.
- Mass is 2,000 kg unless stated (a fixed default, not learned).
- An implied value never fills the quantity being asked for (no implied a when a is the target).
- **Refuses:** two values of the same dimension ("for 14 minutes, or maybe 11 minutes"), or more than two speeds. Two speeds are not a duplicate: they are a start and a final speed.

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
| Number without a unit; unit of a dimension the physics doesn't cover | 2. Extractor |
| Off topic, very ambiguous, more than one question, no question asked | 3. Reader + router |
| Two values of one dimension; more than two speeds | 4. Assembler |
| Inputs contradict each other; missing input; impossible motion (guard) | 5. Planner |
