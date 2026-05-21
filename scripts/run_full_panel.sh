#!/usr/bin/env bash
set -euo pipefail

export UV_NO_SYNC=1

# PyTorch CUDA wheel libraries are under site-packages/nvidia/*/lib after uv sync.
for p in "$PWD"/.venv/lib/python3.11/site-packages/nvidia/*/lib; do
  if [ -d "$p" ]; then
    export LD_LIBRARY_PATH="$p:${LD_LIBRARY_PATH:-}"
  fi
done

# Host NVIDIA driver shim, created by the Nix shell hook.
if [ -d "$PWD/.nix-driver-libs" ]; then
  export LD_LIBRARY_PATH="$PWD/.nix-driver-libs:${LD_LIBRARY_PATH:-}"
fi

CONFIG="${1:-configs/panel.full.yaml}"
shift || true

uv run --no-sync python scripts/run_experiment_panel.py --config "$CONFIG" "$@"
