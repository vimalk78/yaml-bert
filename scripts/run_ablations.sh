#!/bin/bash
# Train all tree-positional-encoding ablation variants in quick mode.
#
# Each variant trains on the same 5K-doc subset for 10 epochs with the same
# seed. Variants differ only in the tree_pos composition (see config.py).
#
# Usage:
#   ./scripts/run_ablations.sh                # default: 5K docs, 10 epochs, seed 42
#   MAX_DOCS=10000 EPOCHS=15 ./scripts/run_ablations.sh
#   SEEDS="42 7" ./scripts/run_ablations.sh   # two-seed variance check
#
# Output: output_ablation_<variant>_seed<n>/ for each (variant, seed) pair.

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

MAX_DOCS="${MAX_DOCS:-5000}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-32}"
SEEDS="${SEEDS:-42}"
VARIANTS="${VARIANTS:-full no_depth no_sibling sequential}"

echo "============================================================"
echo "  YAML-BERT Ablation Sweep"
echo "  Variants: $VARIANTS"
echo "  Seeds:    $SEEDS"
echo "  Docs:     $MAX_DOCS / epochs: $EPOCHS / batch: $BATCH_SIZE"
echo "============================================================"

for variant in $VARIANTS; do
  for seed in $SEEDS; do
    out_dir="output_ablation_${variant}_seed${seed}"
    echo ""
    echo "------------------------------------------------------------"
    echo "  variant=$variant seed=$seed → $out_dir"
    echo "------------------------------------------------------------"
    python scripts/train.py \
      --tree-pos-variant "$variant" \
      --seed "$seed" \
      --max-docs "$MAX_DOCS" \
      --epochs "$EPOCHS" \
      --batch-size "$BATCH_SIZE" \
      --output-dir "$out_dir"
  done
done

echo ""
echo "============================================================"
echo "  All ablation runs complete."
echo "  Evaluate with: python scripts/eval_ablations.py output_ablation_*"
echo "============================================================"
