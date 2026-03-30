# YAML Similarity and Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a document embedding system using pooling attention over the frozen encoder, trained with contrastive loss, enabling similarity search, clustering, and outlier detection.

**Architecture:** A small `DocumentPooling` module (cross-attention layer) sits on top of the frozen encoder. The `kind` node queries all other hidden states to produce a 256-dim document embedding. Trained with supervised contrastive loss using kind labels. Separate checkpoint from the base model.

**Tech Stack:** Python 3.10+, PyTorch, scikit-learn (for clustering)

**Spec:** `docs/superpowers/specs/2026-03-30-yaml-similarity-design.md`

---

### Task 1: DocumentPooling Module

**Files:**
- Create: `yaml_bert/pooling.py`
- Create: `tests/test_pooling.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_pooling.py`

```python
import torch
from yaml_bert.pooling import DocumentPooling


def test_pooling_output_shape():
    d_model = 64
    pooling = DocumentPooling(d_model=d_model, num_heads=4)

    batch_size = 2
    seq_len = 10
    kind_hidden = torch.randn(batch_size, 1, d_model)
    all_hidden = torch.randn(batch_size, seq_len, d_model)

    doc_emb = pooling(kind_hidden, all_hidden)
    assert doc_emb.shape == (batch_size, d_model)


def test_pooling_different_inputs_different_outputs():
    d_model = 64
    pooling = DocumentPooling(d_model=d_model, num_heads=4)

    kind = torch.randn(1, 1, d_model)
    hidden_a = torch.randn(1, 10, d_model)
    hidden_b = torch.randn(1, 10, d_model)

    emb_a = pooling(kind, hidden_a)
    emb_b = pooling(kind, hidden_b)
    assert not torch.allclose(emb_a, emb_b)


def test_pooling_deterministic():
    d_model = 64
    pooling = DocumentPooling(d_model=d_model, num_heads=4)
    pooling.eval()

    kind = torch.randn(1, 1, d_model)
    hidden = torch.randn(1, 10, d_model)

    with torch.no_grad():
        emb1 = pooling(kind, hidden)
        emb2 = pooling(kind, hidden)
    assert torch.equal(emb1, emb2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_pooling.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement DocumentPooling**

File: `yaml_bert/pooling.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DocumentPooling(nn.Module):
    """Pooling by Multi-head Attention.

    The kind node queries all other nodes via cross-attention
    to produce a single document embedding.
    """

    def __init__(self, d_model: int, num_heads: int = 4) -> None:
        super().__init__()
        self.query_proj: nn.Linear = nn.Linear(d_model, d_model)
        self.cross_attn: nn.MultiheadAttention = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True,
        )
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d_model)

    def forward(
        self,
        kind_hidden: torch.Tensor,
        all_hidden: torch.Tensor,
    ) -> torch.Tensor:
        """Produce document embedding from kind node and all hidden states.

        Args:
            kind_hidden: (batch, 1, d_model) — kind node's hidden state
            all_hidden: (batch, seq_len, d_model) — all nodes' hidden states

        Returns:
            (batch, d_model) — document embedding
        """
        query: torch.Tensor = self.query_proj(kind_hidden)
        doc_emb, _ = self.cross_attn(query, all_hidden, all_hidden)
        doc_emb = self.layer_norm(doc_emb)
        return doc_emb.squeeze(1)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Supervised contrastive loss (SupCon).

    Pulls same-label embeddings together, pushes different-label apart.

    Args:
        embeddings: (batch, d_model) — L2-normalized document embeddings
        labels: (batch,) — integer kind labels
        temperature: scaling factor for similarity scores

    Returns:
        scalar loss
    """
    embeddings = F.normalize(embeddings, dim=1)
    batch_size: int = embeddings.shape[0]

    # Pairwise cosine similarity
    sim: torch.Tensor = embeddings @ embeddings.T / temperature

    # Mask: same label = positive pair
    label_mask: torch.Tensor = labels.unsqueeze(0) == labels.unsqueeze(1)

    # Remove self-similarity from positives
    self_mask: torch.Tensor = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
    label_mask = label_mask & ~self_mask

    # For numerical stability
    sim_max, _ = sim.max(dim=1, keepdim=True)
    sim = sim - sim_max.detach()

    # Log-softmax over all non-self entries
    exp_sim: torch.Tensor = torch.exp(sim) * (~self_mask).float()
    log_prob: torch.Tensor = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    # Average over positive pairs
    pos_count: torch.Tensor = label_mask.float().sum(dim=1).clamp(min=1)
    loss: torch.Tensor = -(label_mask.float() * log_prob).sum(dim=1) / pos_count

    return loss.mean()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_pooling.py -v`
Expected: ALL PASS

- [ ] **Step 5: Add contrastive loss test**

Append to `tests/test_pooling.py`:

```python
from yaml_bert.pooling import supervised_contrastive_loss


def test_contrastive_loss_same_labels_lower():
    """Loss should be lower when embeddings match their labels."""
    torch.manual_seed(42)
    # Embeddings that match labels (cluster A close, cluster B close)
    emb_good = torch.tensor([
        [1.0, 0.0], [0.9, 0.1],   # kind 0
        [0.0, 1.0], [0.1, 0.9],   # kind 1
    ])
    labels = torch.tensor([0, 0, 1, 1])

    # Embeddings that don't match (mixed up)
    emb_bad = torch.tensor([
        [1.0, 0.0], [0.0, 1.0],   # kind 0 but one is far
        [0.9, 0.1], [0.1, 0.9],   # kind 1 but one is far
    ])

    loss_good = supervised_contrastive_loss(emb_good, labels)
    loss_bad = supervised_contrastive_loss(emb_bad, labels)

    assert loss_good < loss_bad


def test_contrastive_loss_positive():
    emb = torch.randn(8, 64)
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = supervised_contrastive_loss(emb, labels)
    assert loss.item() > 0
    assert loss.requires_grad
```

- [ ] **Step 6: Run all tests, commit**

Run: `pytest tests/test_pooling.py -v`
Expected: ALL PASS

```bash
git add yaml_bert/pooling.py tests/test_pooling.py
git commit -m "feat: DocumentPooling with cross-attention and supervised contrastive loss"
```

---

### Task 2: Embedding Extraction Helper

Extract hidden states from the frozen encoder and find the kind node position.

**Files:**
- Create: `yaml_bert/similarity.py`
- Create: `tests/test_similarity.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_similarity.py`

```python
import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.similarity import extract_hidden_states, get_document_embedding
from yaml_bert.pooling import DocumentPooling
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder


def _build_model():
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
  - name: app
    image: nginx
""")
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


def test_extract_hidden_states():
    model, vocab = _build_model()
    yaml_text = """\
apiVersion: v1
kind: Pod
metadata:
  name: test
"""
    hidden, kind_pos = extract_hidden_states(model, vocab, yaml_text)

    assert hidden.dim() == 2  # (seq_len, d_model)
    assert hidden.shape[1] == 64
    assert kind_pos >= 0


def test_get_document_embedding():
    model, vocab = _build_model()
    pooling = DocumentPooling(d_model=64, num_heads=2)

    yaml_text = """\
apiVersion: v1
kind: Pod
metadata:
  name: test
"""
    emb = get_document_embedding(model, pooling, vocab, yaml_text)
    assert emb.shape == (64,)


def test_different_yamls_different_embeddings():
    model, vocab = _build_model()
    pooling = DocumentPooling(d_model=64, num_heads=2)

    yaml_a = """\
apiVersion: v1
kind: Pod
metadata:
  name: a
spec:
  containers:
  - name: app
    image: nginx
"""
    yaml_b = """\
apiVersion: v1
kind: Pod
metadata:
  name: b
"""
    emb_a = get_document_embedding(model, pooling, vocab, yaml_a)
    emb_b = get_document_embedding(model, pooling, vocab, yaml_b)
    assert not torch.allclose(emb_a, emb_b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_similarity.py -v`

- [ ] **Step 3: Implement similarity.py**

File: `yaml_bert/similarity.py`

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import _extract_kind
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.pooling import DocumentPooling
from yaml_bert.types import NodeType, YamlNode
from yaml_bert.vocab import Vocabulary


_NODE_TYPE_INDEX: dict[NodeType, int] = {
    NodeType.KEY: 0, NodeType.VALUE: 1,
    NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3,
}


@torch.no_grad()
def extract_hidden_states(
    model: YamlBertModel,
    vocab: Vocabulary,
    yaml_text: str,
) -> tuple[torch.Tensor, int]:
    """Extract encoder hidden states and kind node position.

    Returns:
        (hidden_states: (seq_len, d_model), kind_position: int)
    """
    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()
    nodes: list[YamlNode] = linearizer.linearize(yaml_text)
    if not nodes:
        return torch.empty(0), -1
    annotator.annotate(nodes)

    token_ids: list[int] = []
    node_types: list[int] = []
    depths: list[int] = []
    siblings: list[int] = []
    parent_keys: list[int] = []
    kind_pos: int = -1

    for i, node in enumerate(nodes):
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(_NODE_TYPE_INDEX[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))
        parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))

        if node.token == "kind" and node.depth == 0 and node.node_type == NodeType.KEY and kind_pos == -1:
            kind_pos = i

    kind: str = _extract_kind(nodes)
    kind_id: int = vocab.encode_kind(kind)
    kind_ids: list[int] = [kind_id] * len(nodes)

    t = lambda x: torch.tensor([x])
    model.eval()

    x: torch.Tensor = model.embedding(
        t(token_ids), t(node_types), t(depths), t(siblings), t(parent_keys),
        kind_ids=t(kind_ids),
    )
    for layer in model.encoder.layers:
        x = layer(x)

    return x.squeeze(0), kind_pos


@torch.no_grad()
def get_document_embedding(
    model: YamlBertModel,
    pooling: DocumentPooling,
    vocab: Vocabulary,
    yaml_text: str,
) -> torch.Tensor:
    """Get a single document embedding vector.

    Returns:
        (d_model,) tensor
    """
    hidden, kind_pos = extract_hidden_states(model, vocab, yaml_text)
    if hidden.shape[0] == 0 or kind_pos < 0:
        return torch.zeros(hidden.shape[1] if hidden.dim() > 1 else 1)

    pooling.eval()
    kind_hidden: torch.Tensor = hidden[kind_pos].unsqueeze(0).unsqueeze(0)
    all_hidden: torch.Tensor = hidden.unsqueeze(0)
    return pooling(kind_hidden, all_hidden)


def cosine_similarity_matrix(embeddings: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity.

    Args:
        embeddings: (N, d_model)

    Returns:
        (N, N) similarity matrix
    """
    normed: torch.Tensor = F.normalize(embeddings, dim=1)
    return normed @ normed.T
```

- [ ] **Step 4: Run tests, commit**

Run: `pytest tests/test_similarity.py -v`

```bash
git add yaml_bert/similarity.py tests/test_similarity.py
git commit -m "feat: hidden state extraction and document embedding helpers"
```

---

### Task 3: Pooling Layer Training Script

**Files:**
- Create: `scripts/train_pooling.py`

- [ ] **Step 1: Create training script**

File: `scripts/train_pooling.py`

```python
"""Train the document pooling layer on the frozen encoder.

Usage:
    python scripts/train_pooling.py output_v1/yaml_bert_v1_final.pt --vocab output_v1/vocab.json --epochs 10
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
import time

import torch
from torch.optim import Adam

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.pooling import DocumentPooling, supervised_contrastive_loss
from yaml_bert.similarity import extract_hidden_states
from yaml_bert.vocab import Vocabulary
from yaml_bert.dataset import _extract_kind
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train document pooling layer")
    parser.add_argument("checkpoint", type=str, help="Frozen encoder checkpoint")
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=None, help="Local YAML dir")
    parser.add_argument("--hf-dataset", type=str, default="substratusai/the-stack-yaml-k8s")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--output", type=str, default="pooling_layer.pt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(42)

    print("Loading frozen encoder...")
    vocab: Vocabulary = Vocabulary.load(args.vocab)
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
    cp: dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"], strict=False)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print(f"Encoder frozen (epoch {cp.get('epoch', '?')})")

    # Load documents
    print("Loading documents...")
    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()

    yaml_texts: list[str] = []
    if args.data_dir:
        import glob
        for path in sorted(glob.glob(os.path.join(args.data_dir, "**", "*.yaml"), recursive=True)):
            with open(path) as f:
                yaml_texts.append(f.read())
    else:
        from datasets import load_dataset
        ds = load_dataset(args.hf_dataset, split="train")
        total: int = min(args.max_docs, len(ds))
        yaml_texts = [ds[i]["content"] for i in range(total)]

    # Pre-extract hidden states and kind labels (encoder is frozen)
    print(f"Extracting hidden states from {len(yaml_texts)} documents...")
    all_hidden: list[torch.Tensor] = []
    all_kind_hidden: list[torch.Tensor] = []
    all_kind_labels: list[int] = []
    skipped: int = 0

    for i, yaml_text in enumerate(yaml_texts):
        hidden, kind_pos = extract_hidden_states(model, vocab, yaml_text)
        if hidden.shape[0] == 0 or kind_pos < 0:
            skipped += 1
            continue

        nodes = linearizer.linearize(yaml_text)
        if not nodes:
            skipped += 1
            continue
        annotator.annotate(nodes)
        kind: str = _extract_kind(nodes)
        kind_id: int = vocab.encode_kind(kind)

        all_hidden.append(hidden)
        all_kind_hidden.append(hidden[kind_pos])
        all_kind_labels.append(kind_id)

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{len(yaml_texts)} extracted")

    print(f"Extracted {len(all_hidden)} documents ({skipped} skipped)")

    # Create pooling layer
    pooling: DocumentPooling = DocumentPooling(d_model=args.d_model, num_heads=args.num_heads)
    optimizer: Adam = Adam(pooling.parameters(), lr=args.lr)

    num_params: int = sum(p.numel() for p in pooling.parameters())
    print(f"Pooling layer: {num_params:,} parameters")

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    n: int = len(all_hidden)

    for epoch in range(args.epochs):
        pooling.train()
        # Shuffle
        perm: list[int] = torch.randperm(n).tolist()
        total_loss: float = 0.0
        num_batches: int = 0

        for start in range(0, n, args.batch_size):
            batch_idx: list[int] = perm[start:start + args.batch_size]
            if len(batch_idx) < 2:
                continue

            # Pad hidden states to same length
            batch_hidden: list[torch.Tensor] = [all_hidden[i] for i in batch_idx]
            max_len: int = max(h.shape[0] for h in batch_hidden)
            padded: torch.Tensor = torch.zeros(len(batch_idx), max_len, args.d_model)
            for j, h in enumerate(batch_hidden):
                padded[j, :h.shape[0]] = h

            kind_h: torch.Tensor = torch.stack([all_kind_hidden[i] for i in batch_idx]).unsqueeze(1)
            labels: torch.Tensor = torch.tensor([all_kind_labels[i] for i in batch_idx])

            optimizer.zero_grad()
            doc_embs: torch.Tensor = pooling(kind_h, padded)
            loss: torch.Tensor = supervised_contrastive_loss(doc_embs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        avg_loss: float = total_loss / max(num_batches, 1)
        print(f"Epoch {epoch + 1}/{args.epochs} — loss: {avg_loss:.4f}")

    # Save
    torch.save({
        "pooling_state_dict": pooling.state_dict(),
        "d_model": args.d_model,
        "num_heads": args.num_heads,
    }, args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_pooling.py
git commit -m "feat: pooling layer training script with contrastive loss"
```

---

### Task 4: Clustering and Similarity CLI

**Files:**
- Create: `scripts/cluster_yamls.py`

- [ ] **Step 1: Create CLI**

File: `scripts/cluster_yamls.py`

```python
"""Cluster, search, and find outliers in K8s YAML collections.

Usage:
    python scripts/cluster_yamls.py --encoder ckpt.pt --pooling pooling.pt --corpus ./manifests/ --cluster
    python scripts/cluster_yamls.py --encoder ckpt.pt --pooling pooling.pt --query my.yaml --corpus ./manifests/
    python scripts/cluster_yamls.py --encoder ckpt.pt --pooling pooling.pt --corpus ./manifests/ --outliers
    python scripts/cluster_yamls.py --encoder ckpt.pt --pooling pooling.pt --corpus ./manifests/ --filter-kind Deployment --cluster
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import glob
import os

import torch
import numpy as np

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.pooling import DocumentPooling
from yaml_bert.similarity import get_document_embedding, cosine_similarity_matrix
from yaml_bert.vocab import Vocabulary
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import _extract_kind


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Cluster and search K8s YAMLs")
    parser.add_argument("--encoder", type=str, required=True, help="Encoder checkpoint")
    parser.add_argument("--pooling", type=str, required=True, help="Pooling layer checkpoint")
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--corpus", type=str, required=True, help="Directory of YAML files")
    parser.add_argument("--query", type=str, default=None, help="Find similar to this file")
    parser.add_argument("--cluster", action="store_true", help="Cluster the corpus")
    parser.add_argument("--n-clusters", type=int, default=5, help="Number of clusters")
    parser.add_argument("--outliers", action="store_true", help="Find outliers")
    parser.add_argument("--filter-kind", type=str, default=None, help="Only include this kind")
    parser.add_argument("--top-k", type=int, default=5, help="Top K similar results")
    return parser.parse_args()


def load_models(args) -> tuple[YamlBertModel, DocumentPooling, Vocabulary]:
    torch.manual_seed(42)
    vocab: Vocabulary = Vocabulary.load(args.vocab)
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
    cp = torch.load(args.encoder, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"], strict=False)
    model.eval()

    pooling_cp = torch.load(args.pooling, map_location="cpu", weights_only=False)
    pooling = DocumentPooling(
        d_model=pooling_cp["d_model"],
        num_heads=pooling_cp["num_heads"],
    )
    pooling.load_state_dict(pooling_cp["pooling_state_dict"])
    pooling.eval()

    return model, pooling, vocab


def main() -> None:
    args = parse_args()
    model, pooling, vocab = load_models(args)

    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()

    # Load corpus
    yaml_files: list[str] = sorted(
        glob.glob(os.path.join(args.corpus, "**", "*.yaml"), recursive=True)
    )

    print(f"Embedding {len(yaml_files)} files...")
    embeddings: list[torch.Tensor] = []
    file_names: list[str] = []
    file_kinds: list[str] = []

    for path in yaml_files:
        with open(path) as f:
            yaml_text: str = f.read()

        nodes = linearizer.linearize(yaml_text)
        if not nodes:
            continue
        annotator.annotate(nodes)
        kind: str = _extract_kind(nodes)

        if args.filter_kind and kind != args.filter_kind:
            continue

        emb: torch.Tensor = get_document_embedding(model, pooling, vocab, yaml_text)
        embeddings.append(emb)
        file_names.append(os.path.relpath(path, args.corpus))
        file_kinds.append(kind)

    if not embeddings:
        print("No documents found.")
        return

    all_embs: torch.Tensor = torch.stack(embeddings)
    print(f"Embedded {len(embeddings)} documents")

    # Similarity search
    if args.query:
        with open(args.query) as f:
            query_text: str = f.read()
        query_emb: torch.Tensor = get_document_embedding(model, pooling, vocab, query_text)
        query_nodes = linearizer.linearize(query_text)
        query_kind: str = _extract_kind(query_nodes) if query_nodes else "?"

        sims: torch.Tensor = torch.nn.functional.cosine_similarity(
            query_emb.unsqueeze(0), all_embs,
        )
        top_idx: torch.Tensor = sims.argsort(descending=True)[:args.top_k]

        print(f"\nQuery: {args.query} ({query_kind})")
        print(f"\nMost similar:")
        for rank, idx in enumerate(top_idx):
            i: int = idx.item()
            print(f"  {rank + 1}. [{sims[i]:.3f}] {file_names[i]} ({file_kinds[i]})")

    # Clustering
    if args.cluster:
        from sklearn.cluster import KMeans

        emb_np: np.ndarray = all_embs.numpy()
        n_clusters: int = min(args.n_clusters, len(embeddings))
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels: np.ndarray = kmeans.fit_predict(emb_np)

        print(f"\nClusters (k={n_clusters}):")
        for c in range(n_clusters):
            members: list[int] = [i for i, l in enumerate(labels) if l == c]
            kinds_in_cluster: list[str] = [file_kinds[i] for i in members]
            kind_summary: str = ", ".join(
                f"{k}({kinds_in_cluster.count(k)})"
                for k in sorted(set(kinds_in_cluster))
            )
            print(f"\n  Cluster {c} ({len(members)} docs): {kind_summary}")
            for i in members[:5]:
                print(f"    {file_names[i]} ({file_kinds[i]})")
            if len(members) > 5:
                print(f"    ... and {len(members) - 5} more")

    # Outlier detection
    if args.outliers:
        emb_np = all_embs.numpy()
        centroid: np.ndarray = emb_np.mean(axis=0)
        distances: np.ndarray = np.linalg.norm(emb_np - centroid, axis=1)
        mean_dist: float = distances.mean()
        std_dist: float = distances.std()

        print(f"\nOutliers (>{2:.0f}σ from centroid):")
        outlier_idx: np.ndarray = np.where(distances > mean_dist + 2 * std_dist)[0]
        if len(outlier_idx) == 0:
            print("  No outliers detected.")
        else:
            for i in sorted(outlier_idx, key=lambda x: -distances[x]):
                sigma: float = (distances[i] - mean_dist) / std_dist
                print(f"  [{sigma:.1f}σ] {file_names[i]} ({file_kinds[i]})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add scripts/cluster_yamls.py
git commit -m "feat: cluster_yamls CLI for similarity search, clustering, outlier detection"
```

---

### Task 5: Test End-to-End

- [ ] **Step 1: Train pooling on local data**

```bash
PYTHONPATH=. python scripts/train_pooling.py output_v1/yaml_bert_v1_final.pt \
    --vocab output_v1/vocab.json \
    --data-dir data/k8s-yamls/ \
    --epochs 20 \
    --batch-size 16 \
    --d-model 256 \
    --output output_v1/pooling_layer.pt
```

- [ ] **Step 2: Cluster local YAMLs**

```bash
PYTHONPATH=. python scripts/cluster_yamls.py \
    --encoder output_v1/yaml_bert_v1_final.pt \
    --pooling output_v1/pooling_layer.pt \
    --vocab output_v1/vocab.json \
    --corpus data/k8s-yamls/ \
    --cluster --n-clusters 5
```

- [ ] **Step 3: Find similar to a Deployment**

```bash
PYTHONPATH=. python scripts/cluster_yamls.py \
    --encoder output_v1/yaml_bert_v1_final.pt \
    --pooling output_v1/pooling_layer.pt \
    --vocab output_v1/vocab.json \
    --corpus data/k8s-yamls/ \
    --query data/k8s-yamls/deployment/deployment-nginx.yaml
```

- [ ] **Step 4: Cluster only Deployments**

```bash
PYTHONPATH=. python scripts/cluster_yamls.py \
    --encoder output_v1/yaml_bert_v1_final.pt \
    --pooling output_v1/pooling_layer.pt \
    --vocab output_v1/vocab.json \
    --corpus data/k8s-yamls/ \
    --filter-kind Deployment \
    --cluster --n-clusters 3
```

- [ ] **Step 5: Commit results**

```bash
git add -A
git commit -m "feat: end-to-end YAML similarity and clustering working"
```
