#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-results/smoke_squad_fast_dynamics}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" uv run python run.py \
  --do_train \
  --do_eval \
  --task qa \
  --dataset squad \
  --output_dir "$OUT_DIR" \
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
  --td_dir "$OUT_DIR" \
  --output_dir "$OUT_DIR/cartography"
