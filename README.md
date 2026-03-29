# YAML-BERT

Transformer model with **tree positional encoding** that learns structural patterns from Kubernetes YAML manifests.

## What This Is

Standard transformers use sequential positional encoding (position 1, 2, 3...). YAML is a tree, not a sequence. YAML-BERT replaces sequential encoding with **tree-aware positional encoding** — each node's position is defined by its depth, sibling order, node type, and parent key.

The model learns the statistical regularities of how practitioners write Kubernetes configs — patterns that no schema captures:

- **Structural patterns**: `containers` goes under `spec.template.spec`, not under `metadata`
- **Cross-field correlations**: if `livenessProbe` exists, `readinessProbe` usually exists too
- **Kind-specific structure**: Deployments have `replicas`, Services have `type`

## Architecture

```
YAML Document
    |
    v
YAML Parser → tree of (key, value, depth, parent_path) nodes
    |
    v
Linearize tree → flat sequence of nodes
    |
    v
Token Embedding + Tree Positional Encoding
    |
    v
Transformer Encoder (multi-head self-attention)
    |
    v
Masked key prediction head
```

### Tree Positional Encoding

Six learned embedding tables, all summed into the input vector:

| Component | What it captures |
|-----------|-----------------|
| `key_embedding` | "I am this key" |
| `value_embedding` | "I am this value" |
| `depth_embedding` | "I am at this depth in the tree" |
| `sibling_embedding` | "I am the Nth child of my parent" |
| `node_type_embedding` | "I am a KEY / VALUE / LIST_KEY / LIST_VALUE" |
| `parent_key_embedding` | "My parent is this key" |

## Results (v1.0)

Trained on 276,520 Kubernetes YAML files from [substratusai/the-stack-yaml-k8s](https://huggingface.co/datasets/substratusai/the-stack-yaml-k8s).

| Metric | Value |
|--------|-------|
| Top-1 key prediction accuracy | 95.4% |
| Top-5 key prediction accuracy | 99.4% |
| Capability tests passing | 53/54 (20 capabilities) |
| Model parameters | 7.3M |
| Training | 10 epochs, GTX 1650 |

### Capability Tests

The model passes behavioral tests across 20 capabilities:

- Parent-child validity, Kind conditioning, Depth sensitivity
- Sibling awareness, Required fields, Invalid structure rejection
- Cross-kind discrimination, Volume semantics, StatefulSet/DaemonSet/Job structure
- Probe structure, Security context, RBAC, HPA, Scheduling, and more

## Usage

### Train on local YAML files

```bash
python train.py
```

### Train on HuggingFace dataset

```bash
python train_hf.py --max-docs 1000 --epochs 10           # quick test
python train_hf.py --max-docs 0 --full --epochs 30       # full training
```

### Evaluate a checkpoint

```bash
python evaluate_checkpoint.py output_v1/checkpoints/yaml_bert_epoch_10.pt
```

### Run capability tests

```bash
python test_capabilities.py output_v1/checkpoints/yaml_bert_epoch_10.pt
```

### Anomaly detection

```bash
python anomaly_score.py output_v1/checkpoints/yaml_bert_epoch_10.pt --yaml-file my_manifest.yaml
python anomaly_score.py output_v1/checkpoints/yaml_bert_epoch_10.pt --run-examples
```

### Visualize attention patterns

```bash
python visualize_attention.py output_v1/checkpoints/yaml_bert_epoch_10.pt
```

### Visualize tree embeddings

```bash
python visualize_tree.py output_v1/checkpoints/yaml_bert_epoch_10.pt
```

## Project Structure

```
yaml_bert/
    types.py            # YamlNode, NodeType
    linearizer.py       # YAML to linearized tree nodes
    vocab.py            # Separate key/value vocabularies
    annotator.py        # Domain annotations (list ordering)
    config.py           # Centralized hyperparameters
    embedding.py        # Tree positional encoding (the novel part)
    model.py            # Transformer encoder + key prediction head
    dataset.py          # Dataset with key-only masking
    trainer.py          # Training loop with checkpoint resume
    evaluate.py         # Accuracy, embedding analysis, top-k predictions
    visualize.py        # Loss curves, embedding similarity, attention heatmaps
```

## Documentation

- [Tree Positional Encoding Explained](docs/tree-positional-encoding-explained.md) — mathematical foundations
- [Phase 1 Tokenizer Design](docs/superpowers/specs/2026-03-28-yaml-tokenizer-design.md)
- [Phase 2 Model Design](docs/superpowers/specs/2026-03-28-tree-encoding-model-design.md)
- [Kind Embedding Design](docs/superpowers/specs/2026-03-29-kind-embedding-design.md) (next)

## Requirements

```
pip install -r requirements.txt
```

- Python 3.10+
- PyTorch >= 2.0
- PyYAML
- matplotlib
- datasets (HuggingFace)
