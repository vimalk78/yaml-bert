#!/bin/bash
# Bundle the hf-space/ deployable and upload to the HF Space.
#
# Usage:
#   scripts/deploy_hf_space.sh "commit message"
#
# Copies the canonical yaml_bert/ package, the v9 (subword) checkpoint,
# the BPE tokenizer, and the galaxy data into hf-space/ (all gitignored
# except code), then uploads the whole hf-space/ directory to the Space
# repo via `hf upload`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_DIR="$REPO_ROOT/hf-space"
CHECKPOINT_SRC="$REPO_ROOT/output_v9_276K_recon_seed42/v9_checkpoint.pt"
VOCAB_SRC="$REPO_ROOT/output_v9_276K_recon_seed42/vocab.json"
TOKENIZER_SRC="$REPO_ROOT/tokenizers/v9_unified_bpe_8k.json"
SPACE_REPO="vimalk78/yaml-bert"
COMMIT_MSG="${1:-deploy: bundle update}"

for f in "$CHECKPOINT_SRC" "$VOCAB_SRC" "$TOKENIZER_SRC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required artifact not found at $f" >&2
        exit 1
    fi
done

echo "Staging yaml_bert/ package..."
rm -rf "$SPACE_DIR/yaml_bert"
cp -r "$REPO_ROOT/yaml_bert" "$SPACE_DIR/yaml_bert"
find "$SPACE_DIR/yaml_bert" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

echo "Staging model checkpoint + vocab + tokenizer..."
mkdir -p "$SPACE_DIR/model" "$SPACE_DIR/tokenizers"
cp "$CHECKPOINT_SRC" "$SPACE_DIR/model/yaml_bert.pt"
cp "$VOCAB_SRC" "$SPACE_DIR/model/vocab.json"
# Tokenizer must land where vocab.json's tokenizer_path expects it.
# Our vocab.json stores "tokenizers/v9_unified_bpe_8k.json" — copy
# accordingly so the relative path resolves on the Space.
cp "$TOKENIZER_SRC" "$SPACE_DIR/tokenizers/v9_unified_bpe_8k.json"

echo "Uploading to HF Space $SPACE_REPO..."
cd "$SPACE_DIR"
hf upload "$SPACE_REPO" . --type space --commit-message "$COMMIT_MSG"

echo "Done."
