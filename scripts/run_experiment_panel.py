#!/usr/bin/env python3
"""Run the ELECTRA-small cartography/adversarial QA experiment panel.

This is designed to be launched on the GPU server from inside the Nix+uv shell:

    uv run --no-sync python scripts/run_experiment_panel.py --config configs/panel.full.yaml

The script is resumable. It skips train/eval units whose expected metrics already
exist, writes command/environment logs, saves raw predictions, emits normalized
metrics JSON files, and finally regenerates aggregate tables.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


@dataclass

class TrainSpec:
    run_id: str
    model_short: str
    model_name: str
    train_subset: str
    subset_fraction: float
    subset_draw_id: int | None
    seed: int
    train_budget_type: str
    train_data: str
    output_dir: str
    confidence_definition: str
    num_train_epochs: float
    max_steps: int | None
    save_dynamics: bool

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/panel.full.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--limit-runs", type=int, default=None, help="Debugging aid; do not use for final tables.")
    return parser.parse_args()

def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def fraction_label(frac: float) -> str:
    text = f"{frac:.6f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()

def read_jsonl_count(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count

def run_cmd(args: list[str], *, log_path: Path, dry_run: bool = False, cwd: Path | None = None) -> None:
    cwd = cwd or Path.cwd()
    display = shlex.join(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[{now_utc()}] cwd={cwd}\n$ {display}\n")
    print(f"$ {display}")
    if dry_run:
        return
    start = time.time()
    subprocess.run(args, cwd=str(cwd), check=True)
    elapsed = time.time() - start
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[elapsed_seconds] {elapsed:.1f}\n")
