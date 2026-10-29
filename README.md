# Newton net

Newton net is a tiny language model (about 0.25–0.5M parameters) that reads short driving questions (≤160 characters), picks a simple laws of motion calculation, estimates uncertain inputs (speed, deceleration, reaction time) and runs a fixed, differentiable Newtonian engine inside its own `forward()`. It returns an answer, an uncertainty interval and a physics trace. The neural part interprets the question; the maths is exact.

## Project docs

- [overview.md](./overview.md) — objective, inputs, success criteria and ideas.
- [summary.md](./summary.md) — current project status at a glance.
- [decisions.md](./decisions.md) — design decisions agreed so far.
- [todolist.txt](./todolist.txt) — Todo+ task list (backlog and archive of completed work).
- `history.md` — long-form summary of the 2026-10-29 design chat (local only, git-ignored).

## Other folders

- `data/` — raw source data files (not committed).
- `dataset/` — synthetic training data generator and splits.
- `model/` — TinyLLM, unit normaliser and NewtonPhysics engine.
- `space/` — deployment (Hugging Face endpoint handler / demo).
- `blog/` — write-up.
