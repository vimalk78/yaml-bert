# Missing Field Suggestion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a convention-based linter that uses the pre-trained model to suggest missing fields in Kubernetes YAML manifests.

**Architecture:** For each tree level in a document, mask each existing key and collect the model's top predictions. Keys predicted with high confidence that are absent from the document are reported as missing. No fine-tuning — uses the pre-trained model as-is.

**Tech Stack:** Python 3.10+, PyTorch (inference only)

**Spec:** `docs/superpowers/specs/2026-03-30-missing-field-suggestion-design.md`

---

### Task 1: Core suggestion logic

**Files:**
- Create: `yaml_bert/suggest.py`
- Create: `tests/test_suggest.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_suggest.py`

```python
import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder
from yaml_bert.suggest import suggest_missing_fields


def _build_model_and_vocab():
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    # Build from a representative YAML that has common fields
    yaml_with_all_fields = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  namespace: default
  labels:
    app: web
  annotations:
    description: test
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: web
        image: nginx
        ports:
        - containerPort: 80
        resources:
          limits:
            memory: 128Mi
          requests:
            memory: 64Mi
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
"""
    nodes = linearizer.linearize(yaml_with_all_fields)
    annotator.annotate(nodes)
    vocab = VocabBuilder().build(nodes)

    config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    model = YamlBertModel(
        config=config, embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    return model, vocab


def test_suggest_returns_list():
    model, vocab = _build_model_and_vocab()
    yaml_text = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 3
"""
    suggestions = suggest_missing_fields(model, vocab, yaml_text)
    assert isinstance(suggestions, list)


def test_suggest_each_item_has_required_keys():
    model, vocab = _build_model_and_vocab()
    yaml_text = """\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
"""
    suggestions = suggest_missing_fields(model, vocab, yaml_text, threshold=0.1)
    for s in suggestions:
        assert "parent_path" in s
        assert "missing_key" in s
        assert "confidence" in s
        assert 0.0 <= s["confidence"] <= 1.0


def test_suggest_sorted_by_confidence():
    model, vocab = _build_model_and_vocab()
    yaml_text = """\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
"""
    suggestions = suggest_missing_fields(model, vocab, yaml_text, threshold=0.1)
    if len(suggestions) > 1:
        for i in range(len(suggestions) - 1):
            assert suggestions[i]["confidence"] >= suggestions[i + 1]["confidence"]


def test_suggest_does_not_report_existing_keys():
    model, vocab = _build_model_and_vocab()
    yaml_text = """\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
"""
    suggestions = suggest_missing_fields(model, vocab, yaml_text, threshold=0.01)
    existing_keys_by_parent = {}
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.annotator import DomainAnnotator
    from yaml_bert.types import NodeType
    nodes = YamlLinearizer().linearize(yaml_text)
    DomainAnnotator().annotate(nodes)
    for n in nodes:
        if n.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            existing_keys_by_parent.setdefault(n.parent_path, set()).add(n.token)

    for s in suggestions:
        parent = s["parent_path"]
        if parent in existing_keys_by_parent:
            assert s["missing_key"] not in existing_keys_by_parent[parent], \
                f"Reported '{s['missing_key']}' as missing but it exists at {parent}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_suggest.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement suggest_missing_fields**

File: `yaml_bert/suggest.py`

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import _extract_kind
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.types import NodeType, YamlNode
from yaml_bert.vocab import Vocabulary


_NODE_TYPE_INDEX: dict[NodeType, int] = {
    NodeType.KEY: 0,
    NodeType.VALUE: 1,
    NodeType.LIST_KEY: 2,
    NodeType.LIST_VALUE: 3,
}


def suggest_missing_fields(
    model: YamlBertModel,
    vocab: Vocabulary,
    yaml_text: str,
    threshold: float = 0.3,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Suggest missing fields in a YAML document based on model conventions.

    Args:
        model: Trained YAML-BERT model
        vocab: Vocabulary
        yaml_text: Raw YAML text
        threshold: Minimum confidence to report a missing field
        top_k: Number of predictions per masked position

    Returns:
        List of suggestions sorted by confidence (highest first).
        Each suggestion: {"parent_path": str, "missing_key": str, "confidence": float}
    """
    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()
    nodes: list[YamlNode] = linearizer.linearize(yaml_text)
    if not nodes:
        return []
    annotator.annotate(nodes)

    # Encode all nodes to tensors
    token_ids, node_types, depths, siblings, parent_keys = _encode_nodes(nodes, vocab)

    kind: str = _extract_kind(nodes)
    kind_id: int = vocab.encode_kind(kind)
    kind_ids: list[int] = [kind_id] * len(nodes)

    mask_id: int = vocab.special_tokens["[MASK]"]

    # Group key nodes by parent_path
    keys_by_parent: dict[str, set[str]] = {}
    key_positions_by_parent: dict[str, list[int]] = {}

    for i, node in enumerate(nodes):
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            keys_by_parent.setdefault(node.parent_path, set()).add(node.token)
            key_positions_by_parent.setdefault(node.parent_path, []).append(i)

    # For each parent level, mask each key and collect predictions
    model.eval()
    predicted_keys_by_parent: dict[str, dict[str, float]] = {}

    t = lambda x: torch.tensor([x])

    for parent_path, positions in key_positions_by_parent.items():
        predicted: dict[str, float] = {}

        for pos in positions:
            masked_ids: list[int] = token_ids.copy()
            masked_ids[pos] = mask_id

            with torch.no_grad():
                key_logits, _, _ = model(
                    t(masked_ids), t(node_types), t(depths), t(siblings), t(parent_keys),
                    kind_ids=t(kind_ids),
                )

            probs: torch.Tensor = F.softmax(key_logits[0, pos], dim=-1)
            topk = probs.topk(top_k)

            for j in range(top_k):
                key_name: str = vocab.decode_key(topk.indices[j].item())
                prob: float = topk.values[j].item()
                if key_name in ("[PAD]", "[UNK]", "[MASK]"):
                    continue
                if key_name not in predicted or prob > predicted[key_name]:
                    predicted[key_name] = prob

        predicted_keys_by_parent[parent_path] = predicted

    # Find missing keys
    suggestions: list[dict[str, Any]] = []

    for parent_path, predicted in predicted_keys_by_parent.items():
        existing: set[str] = keys_by_parent.get(parent_path, set())
        for key_name, confidence in predicted.items():
            if key_name not in existing and confidence >= threshold:
                suggestions.append({
                    "parent_path": parent_path,
                    "missing_key": key_name,
                    "confidence": confidence,
                })

    suggestions.sort(key=lambda s: -s["confidence"])
    return suggestions


def _encode_nodes(
    nodes: list[YamlNode],
    vocab: Vocabulary,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    """Encode nodes to integer lists for model input."""
    token_ids: list[int] = []
    node_types: list[int] = []
    depths: list[int] = []
    siblings: list[int] = []
    parent_keys: list[int] = []

    for node in nodes:
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(_NODE_TYPE_INDEX[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))
        parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))

    return token_ids, node_types, depths, siblings, parent_keys
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_suggest.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/suggest.py tests/test_suggest.py
git commit -m "feat: suggest_missing_fields — convention-based missing field detection"
```

---

### Task 2: CLI tool

**Files:**
- Create: `scripts/suggest_fields.py`

- [ ] **Step 1: Create CLI script**

File: `scripts/suggest_fields.py`

```python
"""Suggest missing fields in Kubernetes YAML using YAML-BERT conventions.

Usage:
    python scripts/suggest_fields.py output_v1/yaml_bert_v1_final.pt --yaml-file my-deployment.yaml
    python scripts/suggest_fields.py output_v1/yaml_bert_v1_final.pt --yaml-file my-pod.yaml --threshold 0.5
    python scripts/suggest_fields.py output_v1/yaml_bert_v1_final.pt --yaml-dir ./manifests/
    python scripts/suggest_fields.py output_v1/yaml_bert_v1_final.pt --yaml-file my-pod.yaml --format json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import glob
import json
import os

import torch

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.suggest import suggest_missing_fields
from yaml_bert.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suggest missing fields in K8s YAML")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, default=None, help="Vocab path (auto-detected if not set)")
    parser.add_argument("--yaml-file", type=str, default=None, help="Single YAML file")
    parser.add_argument("--yaml-dir", type=str, default=None, help="Directory of YAML files")
    parser.add_argument("--yaml-text", type=str, default=None, help="Inline YAML text")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--format", type=str, choices=["text", "json"], default="text")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_model(checkpoint_path: str, vocab_path: str, device: str) -> tuple[YamlBertModel, Vocabulary]:
    vocab: Vocabulary = Vocabulary.load(vocab_path)
    config: YamlBertConfig = YamlBertConfig()
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    model = YamlBertModel(
        config=config, embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    checkpoint: dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model, vocab


def print_report(
    suggestions: list[dict],
    source: str = "",
    fmt: str = "text",
) -> None:
    if fmt == "json":
        print(json.dumps({"source": source, "suggestions": suggestions}, indent=2))
        return

    if source:
        print(f"\n{'=' * 60}")
        print(f"  {source}")
        print(f"{'=' * 60}")

    if not suggestions:
        print("  No missing fields detected.")
        return

    # Group by parent_path
    by_parent: dict[str, list[dict]] = {}
    for s in suggestions:
        by_parent.setdefault(s["parent_path"], []).append(s)

    for parent, items in by_parent.items():
        path_display: str = parent if parent else "(root)"
        print(f"\n  {path_display}:")
        for item in items:
            conf: float = item["confidence"]
            strength: str = "STRONG" if conf > 0.8 else "MODERATE" if conf > 0.5 else "WEAK"
            print(f"    [{conf:5.1%}] {item['missing_key']} ({strength})")

    print(f"\n  Total: {len(suggestions)} suggestions")


def main() -> None:
    args = parse_args()

    # Auto-detect vocab path
    vocab_path: str = args.vocab
    if vocab_path is None:
        checkpoint_dir: str = os.path.dirname(args.checkpoint)
        for candidate in [
            os.path.join(checkpoint_dir, "vocab.json"),
            os.path.join(checkpoint_dir, "..", "vocab.json"),
        ]:
            if os.path.exists(candidate):
                vocab_path = candidate
                break
        if vocab_path is None:
            print("Error: could not find vocab.json. Specify --vocab.")
            return

    print(f"Loading model...")
    model, vocab = load_model(args.checkpoint, vocab_path, args.device)

    yaml_files: list[tuple[str, str]] = []

    if args.yaml_file:
        with open(args.yaml_file) as f:
            yaml_files.append((args.yaml_file, f.read()))
    elif args.yaml_dir:
        for path in sorted(glob.glob(os.path.join(args.yaml_dir, "**", "*.yaml"), recursive=True)):
            with open(path) as f:
                yaml_files.append((path, f.read()))
    elif args.yaml_text:
        yaml_files.append(("(inline)", args.yaml_text))
    else:
        print("Provide --yaml-file, --yaml-dir, or --yaml-text")
        return

    all_suggestions: list[dict] = []

    for source, yaml_text in yaml_files:
        suggestions = suggest_missing_fields(
            model, vocab, yaml_text, threshold=args.threshold,
        )
        all_suggestions.extend(suggestions)
        print_report(suggestions, source=source, fmt=args.format)

    if len(yaml_files) > 1 and args.format == "text":
        print(f"\n{'=' * 60}")
        print(f"  Total: {len(all_suggestions)} suggestions across {len(yaml_files)} files")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it runs**

Run: `PYTHONPATH=. python scripts/suggest_fields.py output_v1/yaml_bert_v1_final.pt --vocab output_v1/vocab.json --yaml-text "apiVersion: v1\nkind: Pod\nmetadata:\n  name: test\nspec:\n  containers:\n  - name: app\n    image: nginx\n"`
Expected: prints suggestions

- [ ] **Step 3: Commit**

```bash
git add scripts/suggest_fields.py
git commit -m "feat: suggest_fields CLI tool for convention-based missing field detection"
```

---

### Task 3: Test with real model on real YAMLs

Run the tool against the local K8s YAML samples and verify suggestions are sensible.

- [ ] **Step 1: Run against a minimal Deployment**

```bash
PYTHONPATH=. python scripts/suggest_fields.py output_v1/yaml_bert_v1_final.pt \
    --vocab output_v1/vocab.json \
    --yaml-text "
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: web
        image: nginx
"
```

Expected: suggestions for `resources`, `ports`, possibly `readinessProbe`, `livenessProbe`.

- [ ] **Step 2: Run against a well-configured Deployment**

```bash
PYTHONPATH=. python scripts/suggest_fields.py output_v1/yaml_bert_v1_final.pt \
    --vocab output_v1/vocab.json \
    --yaml-file data/k8s-yamls/deployment/deployment-nginx.yaml
```

Expected: fewer suggestions (the sample has more fields).

- [ ] **Step 3: Run against the full sample directory**

```bash
PYTHONPATH=. python scripts/suggest_fields.py output_v1/yaml_bert_v1_final.pt \
    --vocab output_v1/vocab.json \
    --yaml-dir data/k8s-yamls/ \
    --threshold 0.5
```

Expected: scan all 52 files, report suggestions per file.

- [ ] **Step 4: Commit any fixes needed**

```bash
git add yaml_bert/suggest.py scripts/suggest_fields.py
git commit -m "fix: adjust suggestion logic based on real-world testing"
```
