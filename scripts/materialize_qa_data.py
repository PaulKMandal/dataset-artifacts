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

def normalize_record(record: dict, idx: int | None = None) -> dict:
    answers = record.get("answers", {})
    if isinstance(answers, list):
        answers = {
            "text": [a.get("text", "") for a in answers],
            "answer_start": [int(a.get("answer_start", 0)) for a in answers],
        }
    else:
        answers = {
            "text": list(answers.get("text", [])),
            "answer_start": [int(x) for x in answers.get("answer_start", [])],
        }

    out = {
        "id": str(record.get("id", idx if idx is not None else "")),
        "title": record.get("title", ""),
        "context": record["context"],
        "question": record["question"],
        "answers": answers,
    }
    if idx is not None:
        out["idx"] = int(idx)
    elif "idx" in record and record["idx"] is not None:
        out["idx"] = int(record["idx"])
    return out

def flatten_squad_json(path: Path, *, with_idx: bool = False) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    idx = 0
    for article in payload.get("data", []):
        title = article.get("title", "")
        for paragraph in article.get("paragraphs", []):
            context = paragraph["context"]
            for qa in paragraph.get("qas", []):
                record = {
                    "id": qa["id"],
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "answers": qa.get("answers", []),
                }
                yield normalize_record(record, idx if with_idx else None)
                idx += 1
