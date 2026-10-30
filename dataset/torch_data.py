"""PyTorch Dataset over the JSONL records written by dataset.generate.

    ds = NewtonDataset("data/sample_100.jsonl")
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)

Tokenisation is left to the model: each batch keeps the raw text and character spans, so slot labels
can be projected onto whatever tokeniser is chosen. Missing values are 0 with a mask of 0.
"""

import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from dataset.generate import MOTIONS, SLOTS


class NewtonDataset(Dataset):
    def __init__(self, path):
        self.records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        inputs = r["inputs_si"] or {}
        return {
            "text": r["text"],
            "route_id": r["route_id"],
            "action_id": r["action_id"],
            "motion_id": MOTIONS.index(r["motion"]) if r["motion"] else -100,  # ignored by cross-entropy
            # values as written (SI) for the slot heads, and what the engine ran on after defaults (acceleration signed)
            "slot_si": [r["slots"][s]["si"] if r["slots"][s] else 0.0 for s in SLOTS],
            "slot_mask": [float(r["slots"][s] is not None) for s in SLOTS],
            "slot_spans": {s: r["slots"][s] for s in SLOTS if r["slots"][s]},
            "inputs_si": [inputs.get(s) or 0.0 for s in SLOTS],
            "accel_bucket": r["accel_bucket"] if r["accel_bucket"] is not None else -100,
            "target": r["target"] or 0.0,
            # the loss compares log|answer|: errors count relative to size, so an 86 km journey
            # doesn't swamp a 20 m stop. The sign (negative when braking) comes from the motion type
            "log_target": math.log(abs(r["target"])) if r["target"] else 0.0,
            "target_mask": float(r["target"] is not None),
        }


def collate(items):
    batch = {
        "text": [x["text"] for x in items],
        "slot_spans": [x["slot_spans"] for x in items],
    }
    for key in ("route_id", "action_id", "motion_id", "accel_bucket"):
        batch[key] = torch.tensor([x[key] for x in items], dtype=torch.long)
    for key in ("slot_si", "slot_mask", "inputs_si", "target", "log_target", "target_mask"):
        batch[key] = torch.tensor([x[key] for x in items], dtype=torch.float32)
    return batch
