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
| `unit` | SI unit of the quantity asked for (`m/s`, `s`, `m`, `m/s2`, `N`), which names the target; null when the question is off topic |
| `case` | why it is answered or refused, for evaluation only: `answer`, `missing_input` (a quantity is asked for but not enough is given), `off_topic` |

Training uses `text`, `answer` and `unit` only: which number is which quantity, the sign and any implied values (from
rest, a stop, a wet road) are not labelled, and the model learns them by matching the answer through the planner.

**How it is made:** `dataset/generate.py` fills each question in from a template and a random, consistent physical
scenario. Values are written in units the project's extractor reads, and the answer is the project's planner applied to
the values as written plus what the question implies (a stop means v = 0, from rest u = 0, cruising a = 0, braking on a
road its typical deceleration), so answers are exact. `train` is 5,000 questions (seed 0) and `test` 1,000 (seed 1) less
any whose text is also in `train`. About 80% are answered, 10% ask for something without giving enough, and 10% are
off topic.

**Preliminary:** this matches the project's first, minimal prototype: 13 answerable templates and a few refusals.
More templates and kinds of refusal (ambiguous, contradictory, impossible motion) come as the project develops.

**Limitations:** the questions come from a fixed set of templates, so wording is far narrower than real questions, and
the test split shares templates with train (it tests new numbers and units, not new phrasing). Reaction time and
negation ("I don't want to stop") are deliberately absent. Road-surface decelerations use typical values (dry 7.5,
wet 5.0, icy 2.0 m/s²).
