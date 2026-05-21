# dataset-artifacts

Project by Kaj Bostrom, Jifan Chen, and Greg Durrett. Code by Kaj Bostrom and Jifan Chen. Modified by Paul Mandal.

This package adds a reproducible Nix + uv environment and replaces the slow QA training-dynamics logger with a scalar logger suitable for dataset cartography.

## What is `training_dynamics.jsonl`?

`training_dynamics.jsonl` is the raw training-time log emitted when `run.py` is called with `--save_dynamics`. It is not the final cartography table. Each JSONL row records what the model did on one training feature at one training step/epoch.

For NLI rows, the important fields are:

- `idx`: original training example index
- `epoch`, `step`: when the row was observed
- `confidence`: probability assigned to the gold label
- `correctness`: whether the predicted label matched the gold label
- `label`, `pred_label`

For QA rows, the important fields are:

- `idx`: original SQuAD example index
- `epoch`, `step`: when the row was observed
- `confidence`: average of p(gold start) and p(gold end), matching the old code's definition
- `joint_confidence`: p(gold start) * p(gold end)
- `correctness`: whether predicted start and end exactly matched the feature's gold start/end labels
- `start_position`, `end_position`, `pred_start`, `pred_end`

The previous implementation wrote full `start_prob` and `end_prob` arrays for every QA feature. This package writes scalar fields only, which avoids serializing huge probability vectors.

## What does `dynamics.py` do?

`dynamics.py` converts raw training dynamics into dataset-cartography coordinates:

- confidence: mean confidence across records for an example
- variability: standard deviation of confidence across records
- correctness: mean correctness across records
- region: `Easy-to-learn`, `Ambiguous`, or `Hard-to-learn`

It also writes plots and `categorized_examples.json`.

Example:

```bash
uv run python dynamics.py \
  --td_dir results/squad_smoke \
  --output_dir results/squad_smoke/cartography \
  --confidence_field confidence
```

The output CSV is `cartography_scores.csv`.

## Local setup for LSP / CPU checks

```bash
nix develop .#default
uv sync --extra cpu --group dev
uv run pytest
uv run python -m py_compile run.py helpers.py dynamics.py compare_adversarial.py
```

This gives a CPU-only Python environment for local editing and LSP use.

## GPU server setup

On the GPU server:

```bash
nix develop .#server
uv sync --extra cuda --group dev
uv run python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
PY
```

The CUDA extra uses PyTorch CUDA 12.1 wheels. That is intentional: your server driver reports CUDA 13.0 support, and NVIDIA drivers are backward-compatible with CUDA runtime versions used by PyTorch wheels.

## One-GPU smoke test

```bash
CUDA_VISIBLE_DEVICES=0 uv run python run.py \
  --do_train \
  --do_eval \
  --task qa \
  --dataset squad \
  --output_dir results/smoke_squad_fast_dynamics \
  --overwrite_output_dir \
  --max_train_samples 2048 \
  --max_eval_samples 512 \
  --max_length 128 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 1 \
  --save_only_final_model \
  --save_dynamics \
  --fp16 \
  --report_to none

uv run python dynamics.py \
  --td_dir results/smoke_squad_fast_dynamics \
  --output_dir results/smoke_squad_fast_dynamics/cartography
```

## Laptop to server workflow

Set these once in your local shell or keybind wrapper:

```bash
export DATASET_ARTIFACTS_SERVER='your-server-alias'
export DATASET_ARTIFACTS_REMOTE_DIR='~/dataset-artifacts'
export DATASET_ARTIFACTS_LOCAL_RESULTS_DIR='./server_results'
```

Then run a server command from the laptop:

```bash
scripts/remote_run.sh uv run python run.py \
  --do_train --do_eval --task qa --dataset squad \
  --output_dir results/squad_seed42 \
  --overwrite_output_dir \
  --max_length 384 \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 64 \
  --num_train_epochs 3 \
  --save_only_final_model \
  --save_dynamics \
  --fp16 \
  --seed 42 \
  --report_to none
```

`scripts/remote_run.sh` rsyncs the repo to the server, runs the command inside `nix develop .#server`, and then calls `scripts/pull_results.sh`. By default, result files and metrics are copied back, but model weights are excluded. To pull weights too:

```bash
PULL_MODELS=1 scripts/pull_results.sh
```

## Using this handoff

The handoff ZIP includes the original `.git` history from the uploaded repository plus a new branch and commit. After unzipping:

```bash
cd dataset-artifacts
git status
git log --oneline --decorate --max-count=3
git push -u origin fast-dynamics-nix-uv
```

If GitHub does not attribute the commit to your account, amend the author email to the exact no-reply email shown in GitHub Settings -> Emails:

```bash
git commit --amend --author='PaulKMandal <EXACT_NOREPLY_FROM_GITHUB_SETTINGS>' --no-edit
```

## Notes on validity

The scalar logger preserves the old QA confidence definition by default: average of the gold start/end probabilities. It also logs `joint_confidence`, which is often a better span-level signal. Because SQuAD contexts may create multiple overflow features per raw example, the cartography score for one `idx` may aggregate multiple feature windows. This is documented and should be considered when interpreting example-level regions.
