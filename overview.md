---
title: Newton net
date: 2026-10-29
---

# Objective

Build Newton net: a TinyLLM router plus a fixed Newtonian physics (3 equations of motion) engine in one computation graph. It is an alternative to prevalent LLM tool calling systems and is heavily inspired by Qwenputer where a computer was put inside Qwen3.5 model. It should answer short questions in a car driving context with exact maths, outputing both result and a trace, and abstain on anything out of scope. The focus is not just braking and deceleration: questions also cover accelerating (e.g. time to reach 60 mph from rest) and constant speed (e.g. distance covered in 20 minutes at 70 mph).

# Inputs

- Synthetic scenarios across three motion types (braking, accelerating, constant speed): speed 5–35 m/s, mass 900–2,200 kg, deceleration 1–9 m/s², acceleration 1–5 m/s², reaction time 0.5–2.5 s, journey times up to 2 h, dry/wet/icy surfaces
- 20–50 question templates plus validated paraphrases, ≤160 characters, across a range of complexity and ambiguity scenarios

# Success criteria and metrics

- Route accuracy, answer MAE/RMSE, unsupported recall and false-positive rate on an 800-example test set
- Learning curves at 500, 1k, 2k, 5k, 10k and 20k examples
- Runs on a CPU-only HF endpoint (`intel-spr x2`, scale-to-zero)
- No baseline comparison in this version: it is a proof of concept on simple equations. Benchmarking against alternatives waits for complex scenarios, where simplicity and robustness can be weighed alongside accuracy (see [decisions.md](./decisions.md))


[Project Summary](./summary.md) <br/>
[Decisions](./decisions.md) <br/>
[Todo list](./todolist.txt) <br/>
