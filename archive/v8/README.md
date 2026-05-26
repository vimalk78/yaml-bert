# v8 Archive (read-only, off-path)

Snapshot of YAML-BERT v8 (MLM + reconstruction, 276K atomic-vocab model) taken on 2026-05-27 before the v9 sub-tokenization rewrite. This directory is intentionally not on `PYTHONPATH` — these files are for reference only, not for runtime import.

Mirrors the same pattern used for `archive/v7/`.

To rehydrate v8 locally:

    git checkout <pre-v9-commit-sha> -- yaml_bert/ scripts/train.py

For deployed v8 checkpoints + tokenizer artifacts, see `output_v8_276K_recon_seed42/`.
