---
title: Newton Net
emoji: 🚗
colorFrom: blue
colorTo: gray
sdk: gradio
sdk_version: 6.30.0
python_version: "3.12"
app_file: app.py
pinned: false
license: cc-by-4.0
models:
  - prteek/newton-net
datasets:
  - prteek/newton-net-questions
short_description: Exact driving physics from a question, with a trace
---

Ask one short question (up to 160 characters) about braking, accelerating or driving at a constant speed, such as "doing 70 mph in the rain, how far to stop?". A fixed extractor reads each number and unit and converts it to SI; a reader decides what is asked and how the values fit together (which speed is the start, braking or accelerating, what is implied); a fixed planner chains the equations of motion and F = m·a to the answer. The app shows the answer in everyday units and a trace: the values read, the reader's choices, the known values and each step of the plan in SI. Questions it can't answer are refused.

Everything is a minimal first version, developed further as the project goes: the reader + router is simple keyword rules standing in for the tiny language model until it is trained. The pipeline is the project's code (`model/`, copied next to `app.py` when the Space is published).

Code: [github.com/prteek/newton-net](https://github.com/prteek/newton-net).
