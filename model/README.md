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
and a trace of every step, and refuses anything it can't answer. An alternative to LLM tool calling:
the physics runs inside the model's forward pass, so the reader learns to read questions from whether the physics gets
the answer right.

**Status: work in progress, no trained weights yet.** This repo holds the pipeline as a minimal first version, each
stage just enough for the demo:

| file | stage |
|---|---|
| `extract.py` | 1. Extractor: each number + unit → dimension and SI value (regex over a table of units); the text with values replaced by markers |
| `reader.py` | 2. Reader + router: what is asked and how the values fit together; keyword rules standing in for the TinyLLM |
| `assembler.py` | 3. Assembler: the extracted values and the reader's choices → one value per quantity |
| `planner.py` | 4. Planner: chains the equations of motion to the target; differentiable on torch tensors |

```python
from model.assembler import assemble
from model.extract import extract, mask
from model.planner import solve
from model.reader import read

question = "A car at 30 m/s brakes at 0.7g. How long until it stops?"
quantities = extract(question)
choice = read(mask(question, quantities))
steps, values = solve(assemble(quantities, choice), choice["target"])
print(steps)  # ['t = (v - u)/a = 4.37']
```

- Training data: [prteek/newton-net-questions](https://huggingface.co/datasets/prteek/newton-net-questions)
- Demo (keyword rules in place of the reader until it is trained): [prteek/newton-net](https://huggingface.co/spaces/prteek/newton-net)
- Code and design notes: [github.com/prteek/newton-net](https://github.com/prteek/newton-net), see `decisions.md`
