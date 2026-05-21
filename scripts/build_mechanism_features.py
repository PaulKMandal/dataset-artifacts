#!/usr/bin/env python3
"""Build per-example clean-to-adversarial failure features.

This script aligns raw prediction JSONL files by SQuAD example id. It does not
pretend that train-set cartography scores directly exist for dev/adversarial
items; cartography columns are left blank unless a provided cartography file has
matching example IDs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from argparse import Namespace
from pathlib import Path

from qa_metrics import normalize_answer

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean-predictions", required=True)
    parser.add_argument("--adversarial-predictions", nargs="+", required=True)
    parser.add_argument("--cartography-scores", default=None)
    parser.add_argument("--out", required=True)
    return parser.parse_args()

def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def word_set(text: str) -> set[str]:
    return set(normalize_answer(text).split())

def overlap(a: str, b: str) -> float:
    aw = word_set(a)
    bw = word_set(b)
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)
