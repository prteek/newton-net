---
title: Project decisions
updated: 2026-10-29
---

Recorded from the 2026-10-29 design chat (`history.md`, sections 5–8).

# Scope

- Five routes: `resulting_distance`, `required_[acceleration|deceleration]`, `resulting_time`, `resulting_speed`, `unsupported`. Without an abstain route the soft mixture makes up answers for off-topic questions.
- Motion is not just braking: each question is one of three motion types, braking, accelerating or constant speed, and every route covers each type where it makes sense (e.g. distance to stop, distance to reach 60 mph from rest, distance covered in 20 minutes at 70 mph). Acceleration is signed in the engine: negative when braking.
- Inputs are ≤160-character, Twitter-like questions.
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

- HF Inference Endpoint `intel-spr x2` (2 vCPU / 4 GB, $0.067/hr), 0–1 replicas, scale-to-zero, custom handler. No GPU.
