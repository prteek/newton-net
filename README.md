# Newton net

Newton net answers short driving questions (≤160 characters) about motion with exact physics computed inside the network. A fixed extractor reads each value and unit; a tiny language model (about 0.25–0.5M parameters) decides what is asked and how the values fit together (which speed is the start, braking or accelerating, what is implied); a fixed, differentiable planner chains the equations of motion to the answer. It returns the answer and a trace of each step, or refuses with a reason. It is trained only on questions and their answers, with gradients flowing through the physics. Questions cover braking, accelerating and constant-speed driving.

## Project docs

- [overview.md](./overview.md) — objective, inputs, success criteria and ideas.
- [summary.md](./summary.md) — current project status at a glance.
- [decisions.md](./decisions.md) — design decisions agreed so far.
- [todolist.txt](./todolist.txt) — Todo+ task list (backlog and archive of completed work).
- `history.md` — long-form summary of the 2026-10-29 design chat (local only, git-ignored).

## Other folders

- `data/` — raw source data files (not committed).
- `dataset/` — synthetic question generator (`generate.py`) and PyTorch dataset (`torch_data.py`).
- `model/` — extractor (`extract.py`), readings and assembler (`reading.py`), planner (`planner.py`); the TinyLLM is still to come.
- `space/` — deployment (Hugging Face endpoint handler / demo).
- `blog/` — write-up.
