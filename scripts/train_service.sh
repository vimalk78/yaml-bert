#!/bin/bash
# Auto-resuming training script. Used by systemd service.
# Finds the latest checkpoint and resumes, or starts fresh.
# Stops automatically when training completes (15 epochs).

set -e

PROJECT_DIR="/home/vimal/src/AI-ML/yaml-bert"
OUTPUT_DIR="${1:-$PROJECT_DIR/output_v4}"
VENV="/home/vimal/src/AI-ML/venv-AI-ML/bin/activate"

cd "$PROJECT_DIR"
source "$VENV"

# Check if training already completed
if [ -f "$OUTPUT_DIR/checkpoints/yaml_bert_epoch_15.pt" ]; then
    echo "Training already complete (epoch 15 checkpoint exists). Exiting."
    exit 0
fi

# Find latest checkpoint
LATEST_CHECKPOINT=""
if [ -d "$OUTPUT_DIR/checkpoints" ]; then
    LATEST_CHECKPOINT=$(ls -t "$OUTPUT_DIR/checkpoints"/yaml_bert_v4_epoch_*.pt 2>/dev/null | head -1)
fi

RESUME_FLAG=""
if [ -n "$LATEST_CHECKPOINT" ]; then
    echo "$(date): Resuming from: $LATEST_CHECKPOINT"
    RESUME_FLAG="--resume $LATEST_CHECKPOINT"
else
    echo "$(date): Starting fresh training"
fi

python scripts/train_v4.py \
    --max-docs 0 \
    --epochs 15 \
    --vocab-min-freq 100 \
    --batch-size 16 \
    --output-dir "$OUTPUT_DIR" \
    $RESUME_FLAG

echo "$(date): Training complete"
