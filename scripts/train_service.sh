#!/bin/bash
# Auto-resuming training script. Used by systemd service.
# Finds the latest checkpoint and resumes, or starts fresh.
#
# Usage:
#   ./scripts/train_service.sh [output_dir] [epochs] [batch_size]
#
# Examples:
#   ./scripts/train_service.sh output_v5 30 64
#   ./scripts/train_service.sh  # defaults: output_v4, 15 epochs, batch 24

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${1:-$PROJECT_DIR/output_v4}"
EPOCHS="${2:-15}"
BATCH_SIZE="${3:-24}"

cd "$PROJECT_DIR"

# Activate venv if it exists
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ]; then
    : # already in a venv
fi

# Find latest checkpoint
LATEST_CHECKPOINT=""
if [ -d "$OUTPUT_DIR/checkpoints" ]; then
    LATEST_CHECKPOINT=$(ls -t "$OUTPUT_DIR/checkpoints"/yaml_bert_v4_epoch_*.pt 2>/dev/null | head -1)
fi

# Check if training already completed
FINAL="$OUTPUT_DIR/checkpoints/yaml_bert_v4_epoch_${EPOCHS}.pt"
if [ -f "$FINAL" ]; then
    echo "Training already complete ($FINAL exists). Exiting."
    exit 0
fi

RESUME_FLAG=""
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "$(date): Resuming from: $LATEST_CHECKPOINT"
    RESUME_FLAG="--resume $LATEST_CHECKPOINT"
else
    echo "$(date): Starting fresh training"
fi

PYTHONPATH=. python scripts/train.py \
    --max-docs 0 \
    --epochs "$EPOCHS" \
    --vocab-min-freq 100 \
    --batch-size "$BATCH_SIZE" \
    --output-dir "$OUTPUT_DIR" \
    $RESUME_FLAG

echo "$(date): Training complete"
