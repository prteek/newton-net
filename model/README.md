---
license: cc-by-4.0
language:
  - en
tags:
  - physics
  - neuro-symbolic
  - differentiable-programming
datasets:
  - prteek/newton-net-questions
---

# Newton net

A tiny language model plus a fixed, differentiable Newtonian planner (the equations of motion and F = m·a) in one
computation graph. It answers short driving questions ("doing 70 mph in the rain, how far to stop?") with exact maths
and a trace of every step, and refuses anything it can't answer, with the reason. An alternative to LLM tool calling:
the physics runs inside the model's forward pass, so the reader learns to read questions from whether the physics gets
the answer right.

**Status: work in progress, no trained weights yet.** This repo holds the code written so far for the model:

| file | stage | status |
|---|---|---|
| `extract.py` | 1. Extractor: each number + unit → dimension and SI value (quantulum3 + pint); the text with values replaced by markers | done |
| `reading.py` | 3. Assembler: the extracted values and the reader's choices → one value per quantity; lists every reading for training | done |
| `planner.py` | 4. Planner: shortest chain of guarded equations from the known values to the target, with conflict checks and refusal reasons; differentiable on torch tensors | done |
| TinyLLM encoder | 2. Reader + router (learned): target, speed order, sign, implied values, road-surface \|a\| | not yet |

```python
from model.extract import extract, mask
from model.planner import solve, trace

steps, values, reason = solve({"u": 20.0, "v": 0.0, "a": -5.0}, "s")
print(trace(steps, values))  # ['s = (v² - u²)/2a = 40']
```

- Training data: [prteek/newton-net-questions](https://huggingface.co/datasets/prteek/newton-net-questions)
- Demo (keyword rules in place of the reader until it is trained): [prteek/newton-net](https://huggingface.co/spaces/prteek/newton-net)
- Code and design notes: [github.com/prteek/newton-net](https://github.com/prteek/newton-net), see `decisions.md`
