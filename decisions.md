---
title: Project decisions
updated: 2026-10-29
---

Recorded from the 2026-10-29 design chat (`history.md`, sections 5–8).

# Scope

- Five routes: `resulting_distance`, `required_[acceleration|deceleration]`, `resulting_time`, `resulting_speed`, `unsupported`. Without an abstain route the soft mixture makes up answers for off-topic questions.
- Motion is not just braking: each question is one of three motion types, braking, accelerating or constant speed, and every route covers each type where it makes sense (e.g. distance to stop, distance to reach 60 mph from rest, distance covered in 20 minutes at 70 mph). Acceleration is signed in the engine: negative when braking.
- Reaction time is out of scope: there is no reaction phase, so braking starts immediately and time is a single variable t. Questions that mention a reaction time are not generated.
- Inputs are ≤160-character, Twitter-like questions.
- This version is a proof of concept, not a benchmark: no comparison against baselines (tool-calling LLMs, separately trained classifier + engine). The equations are simple enough that any LLM or large enough network could learn them, so beating one here would prove little. The aim is to show the architecture works and can extend to more complex scenarios. Real benchmarking happens on those complex scenarios, and should weigh the simplicity and robustness this design offers over the alternatives, not just accuracy.
- Decision support only, not a controller: a verified deterministic controller must decide and act.

# Architecture

- Physics runs inside `TinyLLM.forward()` with gradients flowing through it. This is what makes it not tool calling and lets routing be learned from answers.
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


## Architecture

Input >>
    Checker (# of characters, english)
    Model input transformer (Identify inputs and units, identify requested output, refuse based on inputs or outputs or irrelevant query etc.)
    Unit converter
    Planner
    Output formatter


## Input
- Do not accept inputs in textbox beyond 160 characters
- Extract known quantities along with their units and labels or if no inputs are given
- Whether the input contains acceleration or deceleration and apply +/- sign accordingly
- Identify the requested output or if no output is requested

## Routing
- Very ambiguous questions to be routed to refusal
- Duplicate inputs lead to refusal
- Redundant/Extra inputs are handled by planning engine
- Route non english input characters to refusal
- Multiple questions to be routed to refusal
- Missing required inputs should lead to refusal
- No output requested should lead to refusal
- Routing should be probabilistic and part of learning process
- Possible routes:
    - Calculate speed
    - Calculate time
    - Calculate distance
    - Calculate acceleration
    - Calculate Force
    - Refusal

## Compute Planning
- A separate planning engine plans steps required to get to output from given inputs
- Handles redundant/extra inputs
- Identifies uncomputable scenarios and returns helpful message e.g. missing input

## Unit conversion
- For identified unit in input check if unit in a list that converter can handle, then convert quantities to SI units

## Assumptions
- When to assume and when to refuse. User requirements assumes some missing inputs and refuses on
     others, the following rules should guide the training data in learning assumptions and refusals:

  | Missing input                           | Behaviour                                       |
  |-----------------------------------------|-------------------------------------------------|
  | Speed (when not "from rest")            | Refuse                                          |
  | Speed "from rest (or stationary etc.)"  | 0                                               |
  | Acceleration, with a road surface given | Use the surface learnt value                    |
  | Acceleration, with no surface given     | Assume a dry road and use learnt value          |
  | Distance or time the route needs        | Refuse                                          |
  | Vehicle mass                            | Assume 2000kg unless specified exactly          |


## Equations
- Calculate speed: v = u + a * t
- Calculate acceleration: a = F/m or a = (v-u)/t or a = (v^2 - u^2)/(2 * s) or a = 2 * (s - u * t)/t^2
- Calculate distance: s = u * t + 1/2 * a * t^2 or s = (v^2 - u^2)/(2 * a)
- Calculate force: F = m * a
- Calculate time: t = (v-u)/a or t = 2 * s/(u + v) or t = (-u + sqrt(u^2 + 2 * a * s))/a (equivalently t = 2 * s/(u + sqrt(u^2 + 2 * a * s)), which also covers a = 0 → t = s/u)
- Added so that any three of u, v, a, t, s give the rest: v = sqrt(u^2 + 2 * a * s), v = 2 * s/t - u, s = (u + v)/2 * t
- Implemented in `model/planner.py`: a breadth-first search finds the shortest chain of rules from the known quantities to the target, in a fixed rule order so the same question always gets the same plan
- Guards: each rule checks its inputs before it runs (t, s and m positive; a ≠ 0 or u + v ≠ 0 where they divide; u² + 2as ≥ 0, i.e. the car doesn't stop before covering s; a braking car doesn't run past the moment it stops), and every result is checked (t > 0, s ≥ 0, v ≥ 0). A guarded rule is skipped so another can still reach the target; if none can, the guard reasons, or the missing inputs, become the refusal reason
- Tolerances: zero and sign checks treat |x| < 0.001 (SI) as zero, since values read by a trained model are rarely exactly 0. Stopping within 5% of the distance or time asked about counts as reaching it (at v = 0), since stated values are rounded
- Extra inputs are checked, not ignored: before planning, everything the inputs determine is derived and every rule that can recompute a value from the others is compared with it. A gap of more than 5% (or 0.05 in SI near zero) is refused as a conflict naming the value and the rule, e.g. "v is given as 26.82, but v = u + a·t gives 12". A conflict within a single quantity ("stationary at 20 m/s", two speeds for u) can't be seen by the planner; it has to be caught when values are read

## Output
- There are no implausible outputs as this is a proof of concept
- Format output to be user friendly
    - Time output in seconds and if output > 3600 use hours
    - Distance output in m if output > 2000 use km
    - Speed output in km/h always
    - Acceleration output in m/s
    - Force output in N
