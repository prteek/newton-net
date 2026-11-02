---
license: cc-by-4.0
language:
  - en
pretty_name: Newton net driving physics questions
tags:
  - synthetic
  - physics
  - question-answering
  - automotive
size_categories:
  - 1K<n<10K
configs:
  - config_name: default
    data_files:
      - split: train
        path: train.jsonl
      - split: test
        path: test.jsonl
---

# Newton net driving physics questions

Preliminary synthetic data for [Newton net](https://huggingface.co/prteek/newton-net): short (at most 160 characters)
questions about braking, accelerating and driving at a constant speed, each with its exact answer in SI, or no answer
when the question should be refused. Try the questions in the [demo app](https://huggingface.co/spaces/prteek/newton-net).
Code: [github.com/prteek/newton-net](https://github.com/prteek/newton-net).

| column | content |
|---|---|
| `id` | row number within the split |
| `text` | the question |
| `answer` | the answer in SI, signed (negative acceleration or force when braking); null when the question should be refused |
| `unit` | SI unit of the quantity asked for (`m/s`, `s`, `m`, `m/s2`, `N`), which names the target; null when the question isn't one the system answers at all |
| `case` | why it is answered or refused, for evaluation only: `answer`, `off_topic`, `ambiguous`, `multiple_questions`, `no_question`, `bad_unit`, `duplicate`, `missing_input`, `conflict`, `impossible` |

Training uses `text`, `answer` and `unit` only: which number is which quantity, the sign and any implied values (from
rest, a stop, a wet road) are not labelled, and the model learns them by matching the answer through the planner.

**How it is made:** `dataset/generate.py` builds each question from templates and a full, consistent physical scenario,
so answers are exact. Every record is checked against the project's real extractor, readings and planner: an answered
question must have a reading that reproduces its answer, and a refused one must be refused at the right stage.
`train` is 5,000 questions (seed 0) and `test` 1,000 (seed 1) less any whose text is also in `train`. About 60% are
answered; the rest are refusals of each kind.

**Limitations:** the questions come from a fixed set of templates, so wording is far narrower than real questions, and
the test split shares templates with train (it tests new numbers and units, not new phrasing). Reaction time and
negation ("I don't want to stop") are deliberately absent. Road-surface decelerations use typical values (dry 7.5,
wet 5.0, icy 2.0 m/s²).
