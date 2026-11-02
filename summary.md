---
title: Project status
updated at: 2026-11-02
---

# Design

- Newton net design agreed in the 2026-10-29 chat and refined on 2026-10-30 (see [decisions.md](./decisions.md)): extractor → reader + router (learned) → assembler → planner → output
- Trained only on question, answer and unit; the reader + router learns by matching answers through the planner, with the sign and road-surface acceleration trained through the physics
- Model specified apart from the evaluation plan

# Build

- 2026-11-02: cut back to a minimal, demo-first prototype in which every stage is just enough for the demo and fits the others (decisions.md, Prototype):
  - Extractor (`model/extract.py`): regex over a table of units, plus value markers
  - Reader + router (`model/reader.py`): keyword stand-in for the TinyLLM
  - Assembler (`model/assembler.py`): the reader's choices → known values
  - Planner (`model/planner.py`): 10 equations, applied until the target is known
  - Data generator (`dataset/generate.py`): 13 templates plus refusals; answers from the planner on the values as written
  - Space (`space/app.py`): answer or one general refusal, with a trace of each stage
- Not built yet: TinyLLM encoder and heads, training loop, PyTorch dataset
