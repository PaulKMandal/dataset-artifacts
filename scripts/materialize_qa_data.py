#!/usr/bin/env python3
"""Materialize SQuAD and adversarial SQuAD evalsets as flat JSONL.

The experiment panel consumes flat SQuAD-style JSONL so every train/eval file has
an auditable path, checksum, and example count. By default this script downloads:

* SQuAD v1.1 train/validation via ``datasets.load_dataset("squad")``
* AddSent/AddOneSent via ``stanfordnlp/squad_adversarial``

You can also point it at local SQuAD-v1-style JSON files; this is useful when the
server is expected to use a frozen private data mirror.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import datasets

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/qa")
    parser.add_argument("--squad-train-json", default=None)
    parser.add_argument("--squad-dev-json", default=None)
    parser.add_argument("--addsent-json", default=None)
    parser.add_argument("--addonesent-json", default=None)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        default=True,
        help="Pass trust_remote_code=True for stanfordnlp/squad_adversarial.",
    )
    return parser.parse_args()

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()
