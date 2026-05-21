#!/usr/bin/env python3
"""Write a lightweight evalset/table-label audit for the result tree."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


EXPECTED_COUNTS = {
    "squad_dev": 10570,
    "addsent": 3560,
    "addonesent": 1787,
}

def load_metrics(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted((results_dir / "metrics" / "raw").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        row["metrics_file"] = str(path)
        rows.append(row)
    return rows

def group_by_evalset(rows: list[dict]) -> dict[str, list[dict]]:
    by_evalset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_evalset[row.get("evalset", "UNKNOWN")].append(row)
    return by_evalset

def evalset_status(evalset: str, group: list[dict]) -> tuple[list[str], list[str], list[str], str]:
    observed_counts = sorted({str(r.get("num_eval_examples")) for r in group})
    paths = sorted({str(r.get("dataset_path")) for r in group})
    hashes = sorted({str(r.get("dataset_hash")) for r in group})
    expected = EXPECTED_COUNTS.get(evalset)
    status = "OK"
    if expected is not None and observed_counts != [str(expected)]:
        status = "CHECK_COUNT"
    if len(paths) != 1 or len(hashes) != 1:
        status = "CHECK_MULTIPLE_DATASETS"
    return observed_counts, paths, hashes, status
