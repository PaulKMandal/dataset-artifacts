((nil . ((my/remote-host . "rhel-test")
         (my/remote-dir . "/home/rhel/Projects/dataset-artifacts")
         (my/remote-setup-cmd . "nix develop .#server --command bash -lc 'uv sync --frozen --extra cuda --group dev && uv run --no-sync python scripts/materialize_qa_data.py --out-dir data/qa'")
         (my/remote-test-cmd . "nix develop .#server --command bash -lc 'uv run --no-sync pytest'")
         (my/remote-smoke-cmd . "nix develop .#server --command bash -lc 'CUDA_VISIBLE_DEVICES=0 scripts/run_qa_smoke.sh'")
         (my/remote-run-cmd . "nix develop .#server --command bash -lc 'CUDA_VISIBLE_DEVICES=0 scripts/run_full_panel.sh configs/panel.full.yaml'")
         (my/remote-sync-excludes . (".direnv/"
                            ".venv/"
                            "__pycache__/"
                            ".mypy_cache/"
                            ".pytest_cache/"
                            ".ruff_cache/"
                            ".cache/"
                            ".nix-driver-libs/"
                            "wandb/"
                            "data/"
                            "results/"
                            "outputs/"
                            "checkpoint-*/"
                            "*.pyc"
                            "*.pt"
                            "*.bin"
                            "*.safetensors"))
         )))
