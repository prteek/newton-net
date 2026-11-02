---
title: Project status
updated at: 2026-10-30
---

# Design

- Newton net design agreed in the 2026-10-29 chat and refined on 2026-10-30 (see [decisions.md](./decisions.md)): extractor → reader + router (learned) → assembler → planner → output
- Trained only on question, answer and unit; the reader + router learns by matching answers through the planner, with the sign and road-surface acceleration trained through the physics
- Model specified apart from the evaluation plan

# Build

- Planner (`model/planner.py`): full rule set, guards, tolerances, conflict check and missing-input reasons
- Extractor (`model/extract.py`): quantulum3 + pint with fixes for driving questions, plus value markers for the encoder
- Readings and assembler (`model/reading.py`), with soft sign
- Data generator (`dataset/generate.py`) and PyTorch dataset (`dataset/torch_data.py`): every generated question checked against the real extractor, readings and planner
- Not built yet: TinyLLM encoder and heads, training loop, output formatter, Space
