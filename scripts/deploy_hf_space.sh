#!/bin/bash
# Bundle the hf-space/ deployable and upload to the HF Space.
#
# Usage:
#   scripts/deploy_hf_space.sh "commit message"
#
# Copies the canonical yaml_bert/ package and the v7 checkpoint
# into hf-space/ (both gitignored), then uploads the whole hf-space/
# directory to the Space repo via `hf upload`.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SPACE_DIR="$REPO_ROOT/hf-space"
CHECKPOINT_SRC="$REPO_ROOT/output_v7_seed42/checkpoints/yaml_bert_v4_epoch_30.pt"
VOCAB_SRC="$REPO_ROOT/output_v7_seed42/vocab.json"
SPACE_REPO="vimalk78/yaml-bert"
COMMIT_MSG="${1:-deploy: bundle update}"

if [ ! -f "$CHECKPOINT_SRC" ]; then
    echo "ERROR: checkpoint not found at $CHECKPOINT_SRC" >&2
    exit 1
fi
if [ ! -f "$VOCAB_SRC" ]; then
    echo "ERROR: vocab not found at $VOCAB_SRC" >&2
    exit 1
fi

echo "Staging yaml_bert/ package..."
rm -rf "$SPACE_DIR/yaml_bert"
cp -r "$REPO_ROOT/yaml_bert" "$SPACE_DIR/yaml_bert"
find "$SPACE_DIR/yaml_bert" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

echo "Staging model checkpoint + vocab..."
mkdir -p "$SPACE_DIR/model"
cp "$CHECKPOINT_SRC" "$SPACE_DIR/model/yaml_bert.pt"
cp "$VOCAB_SRC" "$SPACE_DIR/model/vocab.json"

echo "Uploading to HF Space $SPACE_REPO..."
cd "$SPACE_DIR"
hf upload "$SPACE_REPO" . --type space --commit-message "$COMMIT_MSG"

echo "Done."
