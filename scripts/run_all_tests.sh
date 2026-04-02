#!/bin/bash
# Run all evaluation tests for a YAML-BERT checkpoint.
#
# Usage:
#   ./scripts/run_all_tests.sh <checkpoint> <vocab>
#
# Examples:
#   ./scripts/run_all_tests.sh output_v5/checkpoints/yaml_bert_v4_epoch_30.pt output_v5/vocab.json
#   ./scripts/run_all_tests.sh output_v4/checkpoints/yaml_bert_v4_epoch_15.pt output_v4/vocab.json

set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

if [ $# -lt 2 ]; then
    echo "Usage: $0 <checkpoint> <vocab>"
    echo "Example: $0 output_v5/checkpoints/yaml_bert_v4_epoch_30.pt output_v5/vocab.json"
    exit 1
fi

CHECKPOINT="$1"
VOCAB="$2"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}."

if [ ! -f "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT"
    exit 1
fi
if [ ! -f "$VOCAB" ]; then
    echo "Vocab not found: $VOCAB"
    exit 1
fi

echo "============================================================"
echo "  YAML-BERT Evaluation Suite"
echo "  Checkpoint: $CHECKPOINT"
echo "  Vocab:      $VOCAB"
echo "============================================================"

echo ""
echo "============================================================"
echo "  1. Unit Tests"
echo "============================================================"
if python -c "import pytest_cov" 2>/dev/null; then
    python -m pytest tests/ --cov=yaml_bert --cov-report=term-missing --cov-config=.coveragerc -q
else
    python -m pytest tests/ -q
fi

echo ""
echo "============================================================"
echo "  2. Capability Tests (93 pre-training + 28 fine-tuning)"
echo "============================================================"
python model_tests/test_capabilities.py "$CHECKPOINT" --vocab "$VOCAB"

echo ""
echo "============================================================"
echo "  3. Structural Tests"
echo "============================================================"
python model_tests/test_structural.py "$CHECKPOINT" --vocab "$VOCAB"

echo ""
echo "============================================================"
echo "  4. Document Similarity"
echo "============================================================"
python scripts/test_similarity.py "$CHECKPOINT" --vocab "$VOCAB"

echo ""
echo "============================================================"
echo "  5. Embedding Structure Analysis"
echo "============================================================"
python scripts/test_embedding_structure.py "$CHECKPOINT" --vocab "$VOCAB"

echo ""
echo "============================================================"
echo "  6. Suggest Fields (nginx deployment)"
echo "============================================================"
python scripts/suggest_fields.py "$CHECKPOINT" --vocab "$VOCAB" --yaml-file data/k8s-yamls/deployment/deployment-nginx.yaml

echo ""
echo "============================================================"
echo "  Done."
echo "============================================================"
