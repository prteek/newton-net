---
title: Project decisions
updated: 2026-10-29
---

Recorded from the 2026-10-29 design chat (`history.md`, sections 5–8).

# Scope

- Five routes: `resulting_distance`, `required_[acceleration|deceleration]`, `resulting_time`, `resulting_speed`, `unsupported`. Without an abstain route the soft mixture makes up answers for off-topic questions.
- Inputs are ≤160-character, Twitter-like questions.
- Decision support only, not a controller: a verified deterministic controller must decide and act.

# Architecture

- Physics runs inside `TinyLLM.forward()` with gradients flowing through it. This is what makes it not tool calling and lets routing be learned from answers.
- TinyLLM: d_model 48, 1–2 layers, 48–64 tokens, 2k–8k vocab (~0.25–0.5M params).
- The model extracts value + unit; a deterministic normaliser converts to SI. Unit conversion is never learned.
- Deceleration as discrete buckets (e.g. 19 from 1–10 m/s²) rather than one continuous head, for interpretability.
- We will focus on different mean levels due to unknown/uncertain inputs but not care about uncertainty or propagating it in the first version
- Action head (answer / refuse) we will either answer or reject a request even in cases of ambiguity
- Layered gates: unsupported prob > 0.5, confidence and entropy thresholds, input validation, output sanity and consistency checks

# Training

- Train jointly: the parser has to learn friction variation from words like "wet road" when not explicitly given
- Start with full supervision (route, slots, units, action, target); answer-only supervision risks shortcuts

# Deployment

- HF Inference Endpoint `intel-spr x2` (2 vCPU / 4 GB, $0.067/hr), 0–1 replicas, scale-to-zero, custom handler. No GPU.
