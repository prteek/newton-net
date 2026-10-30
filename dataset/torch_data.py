"""PyTorch Dataset over the JSONL records written by dataset.generate.

    ds = NewtonDataset("data/sample_100.jsonl")
    loader = DataLoader(ds, batch_size=16, shuffle=True, collate_fn=collate)

Each record is only a question, its answer and the unit of the quantity asked for. The target label
comes from the unit; everything else the reader + router decides (which speed is which, the sign,
implied values) is learned by matching the answer through the planner (model/reading.py).

The extractor runs once per question when the dataset loads (about 6 ms each): each item carries the
extracted quantities, for building readings, and the masked text the encoder reads, with every value
replaced by a marker of its dimension ("A car at <speed1> brakes at <acceleration>").
"""

import json
import math
from pathlib import Path

import torch
from torch.utils.data import Dataset

from dataset.generate import TARGET_UNIT, TARGETS
from model.extract import extract, mask

UNIT_TARGET = {unit: target for target, unit in TARGET_UNIT.items()}


class NewtonDataset(Dataset):
    def __init__(self, path):
        self.records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
        self.quantities = [extract(r["text"]) for r in self.records]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        return {
            "text": r["text"],
            "masked": mask(r["text"], self.quantities[i]),
            "quantities": self.quantities[i],
            "target_id": TARGETS.index(UNIT_TARGET.get(r["unit"], "refuse")),
            # signed answer in SI; the loss compares log|answer| (errors relative to size) and the sign
            "answer": r["answer"] or 0.0,
            "log_answer": math.log(abs(r["answer"])) if r["answer"] else 0.0,
            "answer_mask": float(r["answer"] is not None),
        }


def collate(items):
    batch = {key: [x[key] for x in items] for key in ("text", "masked", "quantities")}
    batch["target_id"] = torch.tensor([x["target_id"] for x in items], dtype=torch.long)
    for key in ("answer", "log_answer", "answer_mask"):
        batch[key] = torch.tensor([x[key] for x in items], dtype=torch.float32)
    return batch
