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

def capture_cmd(args: list[str]) -> str:
    try:
        return subprocess.check_output(args, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:  # noqa: BLE001 - environment logging should not fail the panel
        return f"FAILED {shlex.join(args)}: {exc}"

def environment_lines() -> list[str]:
    lines = [
        f"created_at_utc={now_utc()}",
        f"python={sys.executable}",
        capture_cmd([sys.executable, "--version"]),
        "",
        "uv:",
        capture_cmd(["uv", "--version"]),
        "",
        "nix:",
        capture_cmd(["nix", "--version"]),
        "",
        "nvidia-smi:",
        capture_cmd(["nvidia-smi", "-L"]),
        "",
        "torch:",
        capture_cmd([sys.executable, "-c", "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"]),
        "",
        "environment:",
    ]
    for key in sorted(os.environ):
        if key.startswith(("CUDA", "DATASET_ARTIFACTS", "HF_", "TRANSFORMERS", "UV_")):
            lines.append(f"{key}={os.environ[key]}")
    return lines

def write_environment_logs(results_dir: Path) -> None:
    logs_dir = results_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    (logs_dir / "git_commit.txt").write_text(
        "\n".join([capture_cmd(["git", "rev-parse", "HEAD"]), "", "git status --short:", capture_cmd(["git", "status", "--short"]), ""]),
        encoding="utf-8",
    )
    (logs_dir / "environment.txt").write_text("\n".join(environment_lines()) + "\n", encoding="utf-8")

def model_file_exists(out_dir: Path) -> bool:
    has_config = (out_dir / "config.json").exists()
    has_weights = (out_dir / "model.safetensors").exists() or (out_dir / "pytorch_model.bin").exists()
    return has_config and has_weights

def make_run_id(
    model_short: str,
    subset: str,
    frac: float,
    seed: int,
    budget: str,
    *,
    draw_id: int | None = None,
    confidence_definition: str | None = None,
    primary_confidence_definition: str | None = None,
) -> str:
    parts = [model_short, subset, f"frac{fraction_label(frac)}"]
    if draw_id is not None:
        parts.append(f"draw{draw_id:02d}")
    if confidence_definition and confidence_definition != primary_confidence_definition:
        parts.append(confidence_definition)
    parts.extend([f"seed{seed}", budget])
    return "__".join(parts)

def base_train_args(cfg: dict[str, Any], spec: TrainSpec) -> list[str]:
    training = cfg["training"]
    model = cfg["model"]
    args = [
        sys.executable,
        "run.py",
        "--do_train",
        "--task",
        "qa",
        "--dataset",
        spec.train_data,
        "--model",
        spec.model_name,
        "--output_dir",
        spec.output_dir,
        "--overwrite_output_dir",
        "--max_length",
        str(model["max_seq_length"]),
        "--per_device_train_batch_size",
        str(training["per_device_train_batch_size"]),
        "--per_device_eval_batch_size",
        str(training["per_device_eval_batch_size"]),
        "--learning_rate",
        str(training["learning_rate"]),
        "--warmup_ratio",
        str(training.get("warmup_ratio", 0.0)),
        "--weight_decay",
        str(training.get("weight_decay", 0.0)),
        "--num_train_epochs",
        str(spec.num_train_epochs),
        "--save_only_final_model",
        "--seed",
        str(spec.seed),
        "--report_to",
        "none",
    ]
    if training.get("fp16", False):
        args.append("--fp16")
    if spec.max_steps is not None and spec.max_steps > 0:
        args.extend(["--max_steps", str(spec.max_steps)])
    if spec.save_dynamics:
        args.append("--save_dynamics")
    return args

def train_is_done(out_dir: Path, spec: TrainSpec) -> bool:
    done = (out_dir / "train_metrics.json").exists() and model_file_exists(out_dir)
    if spec.save_dynamics:
        done = done and (
            (out_dir / "training_dynamics.jsonl").exists()
            or any(out_dir.glob("training_dynamics.rank*.jsonl"))
        )
    return done

def write_train_config_copy(cfg: dict[str, Any], spec: TrainSpec) -> None:
    config_copy = Path(cfg["panel"]["results_dir"]) / "configs" / f"{spec.run_id}.yaml"
    config_copy.parent.mkdir(parents=True, exist_ok=True)
    payload = {"train_spec": asdict(spec), "panel_config": cfg}
    config_copy.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

def train_model(cfg: dict[str, Any], spec: TrainSpec, *, log_path: Path, dry_run: bool, resume: bool) -> None:
    out_dir = Path(spec.output_dir)
    if resume and train_is_done(out_dir, spec):
        print(f"[skip train] {spec.run_id}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    write_train_config_copy(cfg, spec)
    run_cmd(base_train_args(cfg, spec), log_path=log_path, dry_run=dry_run)

def eval_args(cfg: dict[str, Any], spec: TrainSpec, eval_path: str, eval_out: Path) -> list[str]:
    model = cfg["model"]
    training = cfg["training"]
    args = [
        sys.executable,
        "run.py",
        "--do_eval",
        "--task",
        "qa",
        "--dataset",
        eval_path,
        "--model",
        spec.output_dir,
        "--output_dir",
        str(eval_out),
        "--overwrite_output_dir",
        "--max_length",
        str(model["max_seq_length"]),
        "--per_device_eval_batch_size",
        str(training["per_device_eval_batch_size"]),
        "--seed",
        str(spec.seed),
        "--report_to",
        "none",
    ]
    if training.get("fp16_eval", training.get("fp16", False)):
        args.append("--fp16")
    return args

def check_eval_outputs(eval_out: Path) -> tuple[Path, Path]:
    metrics_path = eval_out / "eval_metrics.json"
    predictions_path = eval_out / "eval_predictions.jsonl"
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    return metrics_path, predictions_path

def normalized_eval_metrics(
    cfg: dict[str, Any],
    spec: TrainSpec,
    evalset: str,
    eval_path: str,
    metrics_path: Path,
    predictions_dest: Path,
) -> dict[str, Any]:
    results_dir = Path(cfg["panel"]["results_dir"])
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    eval_data_path = Path(eval_path)
    train_data_path = Path(spec.train_data)
    return {
        "run_id": f"{spec.run_id}__{evalset}",
        "train_run_id": spec.run_id,
        "model": spec.model_name,
        "model_short": spec.model_short,
        "train_subset": spec.train_subset,
        "subset_size": read_jsonl_count(train_data_path),
        "subset_fraction": spec.subset_fraction,
        "subset_draw_id": spec.subset_draw_id,
        "seed": spec.seed,
        "train_budget_type": spec.train_budget_type,
        "num_train_epochs": spec.num_train_epochs,
        "max_steps": spec.max_steps,
        "evalset": evalset,
        "num_eval_examples": read_jsonl_count(eval_data_path),
        "exact_match": metrics.get("eval_exact_match"),
        "f1": metrics.get("eval_f1"),
        "eval_script": "run.py/custom_squad_postprocess",
        "dataset_path": str(eval_data_path),
        "dataset_hash": sha256_file(eval_data_path),
        "train_dataset_path": str(train_data_path),
        "train_dataset_hash": sha256_file(train_data_path),
        "predictions_path": str(predictions_dest),
        "config_path": str(results_dir / "configs" / f"{spec.run_id}.yaml"),
        "confidence_definition": spec.confidence_definition,
        "created_at_utc": now_utc(),
    }

def write_eval_metrics(
    cfg: dict[str, Any],
    spec: TrainSpec,
    evalset: str,
    eval_path: str,
    eval_out: Path,
    raw_metrics_path: Path,
) -> None:
    metrics_path, predictions_path = check_eval_outputs(eval_out)
    predictions_dest = Path(cfg["panel"]["results_dir"]) / "predictions" / f"{spec.run_id}__{evalset}.jsonl"
    predictions_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(predictions_path, predictions_dest)
    normalized = normalized_eval_metrics(cfg, spec, evalset, eval_path, metrics_path, predictions_dest)
    raw_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    raw_metrics_path.write_text(json.dumps(normalized, indent=2, sort_keys=True), encoding="utf-8")

def eval_model(
    cfg: dict[str, Any],
    spec: TrainSpec,
    evalset: str,
    eval_path: str,
    *,
    log_path: Path,
    dry_run: bool,
    resume: bool,
) -> None:
    results_dir = Path(cfg["panel"]["results_dir"])
    eval_run_id = f"{spec.run_id}__{evalset}"
    eval_out = results_dir / "evals" / eval_run_id
    raw_metrics_path = results_dir / "metrics" / "raw" / f"{eval_run_id}.json"
    if resume and raw_metrics_path.exists():
        print(f"[skip eval] {eval_run_id}")
        return
    eval_out.mkdir(parents=True, exist_ok=True)
    run_cmd(eval_args(cfg, spec, eval_path, eval_out), log_path=log_path, dry_run=dry_run)
    if not dry_run:
        write_eval_metrics(cfg, spec, evalset, eval_path, eval_out, raw_metrics_path)

def ensure_data(cfg: dict[str, Any], *, log_path: Path, dry_run: bool) -> None:
    data_cfg = cfg["data"]
    required = [Path(v) for v in data_cfg["evalsets"].values()] + [Path(data_cfg["squad_train"])]
    if all(path.exists() for path in required):
        print("[data] required JSONL files already exist")
        return
    args = [sys.executable, "scripts/materialize_qa_data.py", "--out-dir", str(data_cfg["data_dir"])]
    for key, cli in [
        ("squad_train_json", "--squad-train-json"),
        ("squad_dev_json", "--squad-dev-json"),
        ("addsent_json", "--addsent-json"),
        ("addonesent_json", "--addonesent-json"),
    ]:
        if data_cfg.get(key):
            args.extend([cli, str(data_cfg[key])])
    run_cmd(args, log_path=log_path, dry_run=dry_run)

def full_seed42_spec(cfg: dict[str, Any]) -> TrainSpec:
    panel = cfg["panel"]
    model = cfg["model"]
    training = cfg["training"]
    seed = int(cfg["cartography"].get("source_seed", 42))
    run_id = make_run_id(
        model["short_name"],
        "full",
        1.0,
        seed,
        "same_epochs",
        confidence_definition=cfg["cartography"]["primary"],
        primary_confidence_definition=cfg["cartography"]["primary"],
    )
    return TrainSpec(
        run_id=run_id,
        model_short=model["short_name"],
        model_name=model["name_or_path"],
        train_subset="full",
        subset_fraction=1.0,
        subset_draw_id=None,
        seed=seed,
        train_budget_type="same_epochs",
        train_data=cfg["data"]["squad_train"],
        output_dir=str(Path(panel["results_dir"]) / "runs" / run_id),
        confidence_definition=cfg["cartography"]["primary"],
        num_train_epochs=float(training["num_train_epochs"]),
        max_steps=None,
        save_dynamics=True,
    )
