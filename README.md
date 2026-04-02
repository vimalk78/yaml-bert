# YAML-BERT

Transformer model with **tree positional encoding** that learns structural patterns from Kubernetes YAML manifests. Built as a research project to explore how attention behaves on structured (tree) data vs sequential text.

## What This Is

Standard transformers use sequential positional encoding (position 1, 2, 3...). YAML is a tree, not a sequence. YAML-BERT replaces sequential encoding with **tree-aware positional encoding** — each node's position is defined by its depth, sibling order, and node type.

The model uses **hybrid bigram/trigram prediction targets** — predicting `spec::replicas` (parent-aware) instead of just `replicas`, and `Deployment::spec::replicas` (kind-aware) for kind-specific fields. This forces the model to learn both universal K8s structure and kind-specific patterns.

**What the model learns:**

- **Structural patterns**: `containers` goes under `spec.template.spec`, not under `metadata`
- **Cross-field correlations**: if `livenessProbe` exists, `readinessProbe` usually exists too
- **Kind-specific structure**: Deployments have `replicas`, Services have `type`, ConfigMaps have `data`
- **Missing field detection**: suggests fields you likely forgot based on learned patterns

## Results

Trained on 276,520 Kubernetes YAML files from [substratusai/the-stack-yaml-k8s](https://huggingface.co/datasets/substratusai/the-stack-yaml-k8s).

| Metric | Value |
|--------|-------|
| Pre-training capability tests | 93/93 (28 capabilities) |
| Fine-tuning capability tests | 26/28 |
| Document similarity (avg off-diagonal cosine) | 0.46 (target: < 0.70) |
| Model parameters | 7.3M |
| Training | 30 epochs on L4 GPU |

### Capability Tests

The model passes behavioral tests across 28 pre-training capabilities including:

- Parent-child validity, Kind conditioning, Depth sensitivity, Sibling awareness
- Required fields, Cross-kind discrimination, Value-context sensitivity
- RBAC structure, Volume semantics, StatefulSet/DaemonSet/Job/CronJob structure
- Probe structure, Security context, HPA, Scheduling, Ingress, PV/PVC
- Workload controller distinction, ConfigMap vs Secret, Container field completeness

See [full test results](docs/evaluation-results.md).

## Architecture

```
YAML Document
    |
    v
YAML Parser -> tree of (key, value, depth, sibling, node_type) nodes
    |
    v
Linearize tree -> flat sequence of nodes (DFS traversal)
    |
    v
Token Embedding + Tree Positional Encoding (depth + sibling + node_type)
    |
    v
Transformer Encoder (6 layers, 8 heads, d_model=256)
    |
    v
Two prediction heads: simple (bigram) + kind-specific (trigram)
```

### Tree Positional Encoding

Five learned embedding tables, summed into the input vector:

| Component | What it captures |
|-----------|-----------------|
| `key_embedding` | Token identity for keys |
| `value_embedding` | Token identity for values |
| `depth_embedding` | Depth in the YAML tree (0 = root, 1 = top-level, ...) |
| `sibling_embedding` | Position among siblings (0th child, 1st child, ...) |
| `node_type_embedding` | KEY / VALUE / LIST_KEY / LIST_VALUE |

Kind and parent-key awareness come from the prediction targets, not the input — this prevents residual shortcutting (see [architecture doc](docs/architecture.md) for design rationale).

## Usage

### Train

```bash
# Quick test (5K docs, 10 epochs)
PYTHONPATH=. python scripts/train.py --max-docs 5000 --epochs 10 --output-dir output_quick

# Full training (276K docs, 30 epochs)
PYTHONPATH=. python scripts/train.py --max-docs 0 --epochs 30 --batch-size 64 --output-dir output_v5
```

The training script auto-downloads the dataset from HuggingFace on first run.

### Run all evaluations

```bash
./scripts/run_all_tests.sh <checkpoint> <vocab>
# Example:
./scripts/run_all_tests.sh output_v5/checkpoints/yaml_bert_v4_epoch_30.pt output_v5/vocab.json
```

Runs: unit tests, 93 capability tests, 9 structural tests, document similarity, embedding structure analysis, and missing field suggestions.

### Suggest missing fields

```bash
PYTHONPATH=. python scripts/suggest_fields.py <checkpoint> --vocab <vocab> --yaml-file my-deployment.yaml
```

Example output:
```
spec:
    [100.0%] strategy (STRONG)
spec.template.spec.containers.0.resources:
    [ 99.9%] requests (STRONG)
metadata:
    [ 93.1%] labels (STRONG)
```

## Project Structure

```
yaml_bert/               # Core library
    types.py             # YamlNode, NodeType
    linearizer.py        # YAML string -> linearized tree nodes
    vocab.py             # Key/value/target vocabularies with hybrid targets
    annotator.py         # Domain annotations (list ordering)
    config.py            # Hyperparameters (d_model=256, layers=6, heads=8)
    embedding.py         # Tree positional encoding
    model.py             # Transformer encoder + dual prediction heads
    dataset.py           # Dataset with key-only masking
    trainer.py           # Training loop with NaN skip and gradient clipping
    suggest.py           # Missing field suggestion tool
    similarity.py        # Document embedding extraction
    pooling.py           # Attention pooling for document embeddings

scripts/                 # Training and evaluation scripts
    train.py             # Main training script
    train_service.sh     # Auto-resume training for systemd
    suggest_fields.py    # CLI for missing field suggestions
    test_similarity.py   # Document similarity matrix
    test_embedding_structure.py  # Learned embedding analysis
    run_all_tests.sh     # Run all evaluations

model_tests/             # Behavioral tests
    test_capabilities.py # 93 capability tests (CheckList methodology)
    test_structural.py   # 9 structural tests

tests/                   # Unit tests (81% coverage on core code)
testdata/                # Sample K8s YAMLs for testing
```

## Documentation

- [Tree Positional Encoding Explained](docs/tree-positional-encoding-explained.md) — mathematical foundations
- [Architecture](docs/architecture.md) — design decisions
- [Evaluation Results](docs/evaluation-results.md) — capability tests, similarity, TPE verification
- [Next Training Improvements](docs/next-training-improvements.md) — planned experiments
- [Development Plans](docs/superpowers/) — AI-assisted planning docs (built with Claude Code)

## Requirements

```
pip install -r requirements.txt
```

- Python 3.10+
- PyTorch >= 2.0
- PyYAML, tqdm, datasets (HuggingFace)

## Built With

This project was built with [Claude Code](https://claude.ai/code) as an AI-assisted development experiment. The [development plans](docs/superpowers/) show how AI was used for architecture design, planning, and implementation.

## License

MIT — see [LICENSE](LICENSE).
