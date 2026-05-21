#!/usr/bin/env python3
"""Create persistent QA subset JSONL files from cartography scores.

Selection rules are intentionally simple and auditable:

* easy: highest confidence, then lowest variability, then highest correctness
* hard: lowest confidence, then lowest correctness, then highest variability
* ambiguous: highest variability, then middle confidence preference
* random: uniform sample without replacement, one persistent file per draw

The script writes a manifest plus a subset_assignments.csv that records exactly
which original SQuAD train indices were selected for every subset/fraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from argparse import Namespace
from pathlib import Path
from typing import Iterable


CARTO_SUBSETS = ("easy", "ambiguous", "hard")

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True, help="Flat SQuAD train JSONL with stable idx fields.")
    parser.add_argument("--cartography-scores", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--assignments-out", default=None)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.333])
    parser.add_argument("--random-draws", type=int, default=10)
    parser.add_argument("--random-seed-base", type=int, default=7300)
    parser.add_argument("--confidence-definition", default="joint_confidence")
    parser.add_argument("--rounding", choices=["round", "floor", "ceil"], default="round")
    return parser.parse_args()

def fraction_label(frac: float) -> str:
    text = f"{frac:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(rows: Iterable[dict], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1
    return count

def read_scores(path: Path) -> dict[int, dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        scores = {}
        for row in reader:
            idx = int(row["idx"])
            scores[idx] = {
                "idx": idx,
                "confidence": float(row.get("confidence", row.get("avg_confidence", 0.0))),
                "variability": float(row.get("variability", 0.0)),
                "correctness": float(row.get("correctness", 0.0)),
                "region": row.get("region", ""),
                "n_records": int(float(row.get("n_records", 0) or 0)),
            }
    if not scores:
        raise SystemExit(f"No cartography scores loaded from {path}")
    return scores

def answer_summary(example: dict) -> tuple[str, str, str]:
    answers = example.get("answers", {})
    texts = answers.get("text", []) if isinstance(answers, dict) else []
    starts = answers.get("answer_start", []) if isinstance(answers, dict) else []
    answer_text = texts[0] if texts else ""
    answer_start = starts[0] if starts else ""
    answer_length = len(answer_text.split()) if answer_text else 0
    return answer_text, str(answer_start), str(answer_length)
