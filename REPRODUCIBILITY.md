# Reproducibility plan

## Environment

- Nix flake provides the system shell: Python 3.11, uv, git, rsync, OpenSSH, and common build libraries.
- uv manages Python dependencies.
- Use `uv sync --extra cpu --group dev` locally.
- Use `uv sync --extra cuda --group dev` on the V100 server.

## Expected commands

Local checks:

```bash
nix develop .#default
uv sync --extra cpu --group dev
uv run pytest
uv run python -m py_compile run.py helpers.py dynamics.py compare_adversarial.py
```

Server smoke test:

```bash
nix develop .#server
uv sync --extra cuda --group dev
CUDA_VISIBLE_DEVICES=0 uv run python run.py \
  --do_train --do_eval --task qa --dataset squad \
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
```

Cartography after training:

```bash
uv run python dynamics.py \
  --td_dir results/smoke_squad_fast_dynamics \
  --output_dir results/smoke_squad_fast_dynamics/cartography
```

## Artifacts to retain

Keep these:

- `eval_metrics.json`
- `eval_predictions.jsonl`
- `training_dynamics.jsonl` or `training_dynamics.rank*.jsonl`
- `cartography/cartography_scores.csv`
- `cartography/categorized_examples.json`
- `cartography/*.png`
- `trainer_state.json`
- command logs and environment logs

Do not copy back full model weights by default unless `PULL_MODELS=1` is set.

## Pass/fail checks

- `--max_length` must affect QA tokenized sequence length.
- `training_dynamics.jsonl` must contain scalar fields, not full `start_prob`/`end_prob` arrays.
- One V100 smoke run should produce non-empty dynamics and eval metrics.
- Cartography script should produce a non-empty `cartography_scores.csv`.
