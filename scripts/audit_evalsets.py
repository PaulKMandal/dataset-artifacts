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
