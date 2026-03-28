# YAML-BERT Phase 2: Tree Positional Encoding and Model — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the full YAML-BERT model with tree positional encoding, transformer encoder, masked key prediction, and training loop.

**Architecture:** Six embedding tables (key, value, depth, sibling, node_type, parent_key) summed into input vectors, fed through a standard TransformerEncoder, with a single linear prediction head for masked key prediction. Only keys are masked; values serve as unmasked context.

**Tech Stack:** Python 3.10+, PyTorch >= 2.0, PyYAML, pytest

**Spec:** `docs/superpowers/specs/2026-03-28-tree-encoding-model-design.md`

---

### Task 1: Add PyTorch Dependency

**Files:**
- Modify: `requirements.txt`

- [ ] **Step 1: Update requirements.txt**

File: `requirements.txt`

```
pyyaml>=6.0
pytest>=7.0
torch>=2.0
```

- [ ] **Step 2: Install and verify**

Run: `pip install -r requirements.txt`
Run: `python -c "import torch; print(f'PyTorch {torch.__version__} OK')"`
Expected: `PyTorch <version> OK`

- [ ] **Step 3: Commit**

```bash
git add requirements.txt
git commit -m "chore: add PyTorch dependency"
```

---

### Task 2: YamlBertConfig Dataclass

Central configuration for all hyperparameters. Change values in one place.

**Files:**
- Create: `yaml_bert/config.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_config.py`

```python
from yaml_bert.config import YamlBertConfig


def test_default_config():
    config = YamlBertConfig()

    assert config.d_model == 256
    assert config.num_layers == 6
    assert config.num_heads == 8
    assert config.d_ff == 1024
    assert config.max_depth == 16
    assert config.max_sibling == 32
    assert config.mask_prob == 0.15
    assert config.lr == 1e-4
    assert config.batch_size == 32
    assert config.num_epochs == 30
    assert config.max_seq_len == 512


def test_custom_config():
    config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)

    assert config.d_model == 64
    assert config.num_layers == 2
    assert config.num_heads == 2
    # d_ff follows d_model by default
    assert config.d_ff == 256


def test_d_ff_defaults_to_4x_d_model():
    config = YamlBertConfig(d_model=128)
    assert config.d_ff == 512

    config_custom = YamlBertConfig(d_model=128, d_ff=1024)
    assert config_custom.d_ff == 1024
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement YamlBertConfig**

File: `yaml_bert/config.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class YamlBertConfig:
    """Central configuration for YAML-BERT hyperparameters."""

    # Model architecture
    d_model: int = 256
    num_layers: int = 6
    num_heads: int = 8
    d_ff: int = 0  # 0 means "auto: 4 * d_model"
    max_depth: int = 16
    max_sibling: int = 32

    # Training
    mask_prob: float = 0.15
    lr: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 30
    max_seq_len: int = 512

    def __post_init__(self) -> None:
        if self.d_ff == 0:
            self.d_ff = 4 * self.d_model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/config.py tests/test_config.py
git commit -m "feat: YamlBertConfig dataclass for centralized hyperparameters"
```

---

### Task 3: Vocabulary — Add parent_key_id Extraction

The embedding layer needs `parent_key_id` for each node — the last non-numeric component of `parent_path`, encoded as a key vocab ID. Add a helper method to `Vocabulary`.

**Files:**
- Modify: `yaml_bert/vocab.py`
- Create: `tests/test_vocab_parent_key.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_vocab_parent_key.py`

```python
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import Vocabulary, VocabBuilder


def test_extract_parent_key():
    vocab = VocabBuilder().build([])

    assert vocab.extract_parent_key("spec.template.spec.containers.0") == "containers"
    assert vocab.extract_parent_key("metadata") == "metadata"
    assert vocab.extract_parent_key("spec.containers.0.ports.1") == "ports"
    assert vocab.extract_parent_key("") == ""
    assert vocab.extract_parent_key("spec") == "spec"
    assert vocab.extract_parent_key("spec.containers.0") == "containers"
    assert vocab.extract_parent_key("args.0") == "args"


def test_encode_parent_key():
    yaml_str = """\
spec:
  replicas: 3
status:
  replicas: 2
"""
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    builder = VocabBuilder()
    vocab = builder.build(nodes)

    # The "replicas" key under "spec" has parent_path="spec"
    # parent_key is "spec", which is in key_vocab
    replicas_under_spec = nodes[1]  # replicas key
    parent_key = vocab.extract_parent_key(replicas_under_spec.parent_path)
    assert parent_key == "spec"
    parent_key_id = vocab.encode_key(parent_key)
    assert parent_key_id != vocab.special_tokens["[UNK]"]

    # The "replicas" key under "status" has parent_path="status"
    replicas_under_status = nodes[3]  # replicas key
    parent_key = vocab.extract_parent_key(replicas_under_status.parent_path)
    assert parent_key == "status"
    parent_key_id = vocab.encode_key(parent_key)
    assert parent_key_id != vocab.special_tokens["[UNK]"]


def test_encode_parent_key_root_nodes():
    """Root-level nodes have parent_path='', parent_key should map to [UNK]."""
    yaml_str = """\
apiVersion: v1
"""
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    builder = VocabBuilder()
    vocab = builder.build(nodes)

    api_key = nodes[0]  # apiVersion key
    parent_key = vocab.extract_parent_key(api_key.parent_path)
    assert parent_key == ""
    # Empty string is not in key_vocab, so encode_key returns [UNK]
    parent_key_id = vocab.encode_key(parent_key)
    assert parent_key_id == vocab.special_tokens["[UNK]"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_vocab_parent_key.py -v`
Expected: FAIL with `AttributeError: 'Vocabulary' object has no attribute 'extract_parent_key'`

- [ ] **Step 3: Add extract_parent_key to Vocabulary**

Add this method to the `Vocabulary` class in `yaml_bert/vocab.py`:

```python
    @staticmethod
    def extract_parent_key(parent_path: str) -> str:
        """Extract the last non-numeric component from a parent_path.

        Examples:
            "spec.template.spec.containers.0" -> "containers"
            "metadata" -> "metadata"
            "" -> ""
        """
        if not parent_path:
            return ""
        parts = parent_path.split(".")
        for part in reversed(parts):
            if not part.isdigit():
                return part
        return ""
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_vocab_parent_key.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run all existing tests to verify no regressions**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add yaml_bert/vocab.py tests/test_vocab_parent_key.py
git commit -m "feat: add extract_parent_key to Vocabulary"
```

---

### Task 4: YamlBertEmbedding

The core novel component — six embedding tables summed into input vectors.

**Files:**
- Create: `yaml_bert/embedding.py`
- Create: `tests/test_embedding.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_embedding.py`

```python
import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.types import NodeType


def _make_embedding(d_model: int = 64, key_vocab: int = 100, val_vocab: int = 200) -> YamlBertEmbedding:
    config = YamlBertConfig(d_model=d_model)
    return YamlBertEmbedding(
        config=config,
        key_vocab_size=key_vocab,
        value_vocab_size=val_vocab,
    )


def test_embedding_output_shape():
    d_model = 64
    emb = _make_embedding(d_model=d_model)

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)  # all KEY
    depths = torch.randint(0, 5, (batch_size, seq_len))
    sibling_indices = torch.randint(0, 3, (batch_size, seq_len))
    parent_key_ids = torch.randint(0, 50, (batch_size, seq_len))

    output = emb(token_ids, node_types, depths, sibling_indices, parent_key_ids)

    assert output.shape == (batch_size, seq_len, d_model)


def test_embedding_routes_by_node_type():
    """KEY/LIST_KEY use key_embedding, VALUE/LIST_VALUE use value_embedding."""
    emb = _make_embedding(d_model=32, key_vocab=10, val_vocab=10)

    token_ids = torch.tensor([[5, 5]])  # same token ID
    depths = torch.tensor([[0, 0]])
    siblings = torch.tensor([[0, 0]])
    parent_keys = torch.tensor([[0, 0]])

    # NodeType.KEY = 0, NodeType.VALUE = 1
    node_types_key = torch.tensor([[0, 0]])
    node_types_value = torch.tensor([[1, 1]])

    out_key = emb(token_ids, node_types_key, depths, siblings, parent_keys)
    out_value = emb(token_ids, node_types_value, depths, siblings, parent_keys)

    # Same token ID but different node_type should produce different vectors
    # (different embedding tables + different node_type_embedding)
    assert not torch.allclose(out_key, out_value)


def test_different_parent_keys_produce_different_embeddings():
    """Two nodes differing only in parent_key should get different embeddings."""
    emb = _make_embedding(d_model=32, key_vocab=10, val_vocab=10)

    token_ids = torch.tensor([[3]])
    node_types = torch.tensor([[0]])  # KEY
    depths = torch.tensor([[1]])
    siblings = torch.tensor([[0]])

    parent_a = torch.tensor([[4]])  # parent_key_id = 4
    parent_b = torch.tensor([[5]])  # parent_key_id = 5

    out_a = emb(token_ids, node_types, depths, siblings, parent_a)
    out_b = emb(token_ids, node_types, depths, siblings, parent_b)

    assert not torch.allclose(out_a, out_b)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedding.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement YamlBertEmbedding**

File: `yaml_bert/embedding.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn

from yaml_bert.config import YamlBertConfig
from yaml_bert.types import NodeType


class YamlBertEmbedding(nn.Module):
    """Embedding layer with tree positional encoding for YAML-BERT.

    Produces input vectors by summing:
    - Token embedding (key_embedding or value_embedding, routed by node_type)
    - Tree positional encoding (depth + sibling + node_type + parent_key)
    """

    def __init__(
        self,
        config: YamlBertConfig,
        key_vocab_size: int,
        value_vocab_size: int,
    ) -> None:
        super().__init__()

        d: int = config.d_model

        # Token embeddings — separate tables for keys and values
        self.key_embedding: nn.Embedding = nn.Embedding(key_vocab_size, d)
        self.value_embedding: nn.Embedding = nn.Embedding(value_vocab_size, d)

        # Tree positional encoding components
        self.depth_embedding: nn.Embedding = nn.Embedding(config.max_depth, d)
        self.sibling_embedding: nn.Embedding = nn.Embedding(config.max_sibling, d)
        self.node_type_embedding: nn.Embedding = nn.Embedding(4, d)
        self.parent_key_embedding: nn.Embedding = nn.Embedding(key_vocab_size, d)

        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d)

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        parent_key_ids: torch.Tensor,
    ) -> torch.Tensor:
        # Route token embedding based on node_type
        # KEY=0, LIST_KEY=2 use key_embedding; VALUE=1, LIST_VALUE=3 use value_embedding
        is_key: torch.Tensor = (node_types == 0) | (node_types == 2)

        key_emb: torch.Tensor = self.key_embedding(token_ids)
        val_emb: torch.Tensor = self.value_embedding(token_ids)
        token_emb: torch.Tensor = torch.where(
            is_key.unsqueeze(-1), key_emb, val_emb
        )

        # Tree positional encoding
        tree_pos: torch.Tensor = (
            self.depth_embedding(depths)
            + self.sibling_embedding(sibling_indices)
            + self.node_type_embedding(node_types)
            + self.parent_key_embedding(parent_key_ids)
        )

        return self.layer_norm(token_emb + tree_pos)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_embedding.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/embedding.py tests/test_embedding.py
git commit -m "feat: YamlBertEmbedding with tree positional encoding"
```

---

### Task 5: YamlBertModel

Wraps embedding + TransformerEncoder + key prediction head.

**Files:**
- Create: `yaml_bert/model.py`
- Create: `tests/test_model.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_model.py`

```python
import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel


# Small config for fast tests
TEST_CONFIG: YamlBertConfig = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)
KEY_VOCAB_SIZE: int = 100
VALUE_VOCAB_SIZE: int = 200


def _make_model() -> YamlBertModel:
    emb = YamlBertEmbedding(
        config=TEST_CONFIG,
        key_vocab_size=KEY_VOCAB_SIZE,
        value_vocab_size=VALUE_VOCAB_SIZE,
    )
    return YamlBertModel(
        config=TEST_CONFIG,
        embedding=emb,
        key_vocab_size=KEY_VOCAB_SIZE,
    )


def test_model_output_shape():
    model = _make_model()

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))

    key_logits = model(token_ids, node_types, depths, siblings, parent_keys)

    assert key_logits.shape == (batch_size, seq_len, KEY_VOCAB_SIZE)


def test_model_with_padding_mask():
    model = _make_model()

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))

    padding_mask = torch.tensor([
        [False, False, False, False, False],
        [False, False, False, True, True],
    ])

    key_logits = model(
        token_ids, node_types, depths, siblings, parent_keys,
        padding_mask=padding_mask,
    )

    assert key_logits.shape == (batch_size, seq_len, KEY_VOCAB_SIZE)


def test_model_loss_computation():
    model = _make_model()

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))

    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels[0, 1] = 10
    labels[1, 0] = 20

    key_logits = model(token_ids, node_types, depths, siblings, parent_keys)
    loss = model.compute_loss(key_logits, labels)

    assert loss.dim() == 0  # scalar
    assert loss.item() > 0  # loss should be positive
    assert loss.requires_grad  # should be differentiable
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_model.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement YamlBertModel**

File: `yaml_bert/model.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding


class YamlBertModel(nn.Module):
    """YAML-BERT: Transformer encoder with tree positional encoding.

    Takes linearized YAML node sequences, applies tree-aware embeddings,
    processes through a transformer encoder, and predicts masked keys.
    """

    def __init__(
        self,
        config: YamlBertConfig,
        embedding: YamlBertEmbedding,
        key_vocab_size: int,
    ) -> None:
        super().__init__()

        self.embedding: YamlBertEmbedding = embedding

        encoder_layer: nn.TransformerEncoderLayer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.num_heads,
            dim_feedforward=config.d_ff,
            batch_first=True,
        )
        self.encoder: nn.TransformerEncoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_layers
        )

        self.key_prediction_head: nn.Linear = nn.Linear(config.d_model, key_vocab_size)
        self.loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        parent_key_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x: torch.Tensor = self.embedding(
            token_ids, node_types, depths, sibling_indices, parent_key_ids
        )
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        key_logits: torch.Tensor = self.key_prediction_head(x)
        return key_logits

    def compute_loss(
        self,
        key_logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # Reshape for CrossEntropyLoss: (batch * seq_len, vocab_size) vs (batch * seq_len,)
        return self.loss_fn(
            key_logits.view(-1, key_logits.size(-1)),
            labels.view(-1),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_model.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/model.py tests/test_model.py
git commit -m "feat: YamlBertModel with transformer encoder and key prediction head"
```

---

### Task 6: YamlDataset — Node Encoding

Convert `YamlNode` lists into tensor-ready integer arrays (before masking).

**Files:**
- Create: `yaml_bert/dataset.py`
- Create: `tests/test_dataset.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_dataset.py`

```python
import os

import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.dataset import YamlDataset
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder
from yaml_bert.types import NodeType

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "k8s-yamls"
)


def _build_vocab():
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    import glob
    all_nodes = []
    for path in glob.glob(os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True):
        nodes = linearizer.linearize_file(path)
        annotator.annotate(nodes)
        all_nodes.extend(nodes)
    return VocabBuilder().build(all_nodes)


def test_dataset_length():
    vocab = _build_vocab()
    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
    )
    assert len(dataset) > 40  # at least 40+ YAML files


def test_dataset_item_keys():
    vocab = _build_vocab()
    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
    )
    item = dataset[0]

    expected_keys = {
        "token_ids", "node_types", "depths",
        "sibling_indices", "parent_key_ids", "labels",
    }
    assert set(item.keys()) == expected_keys

    # All tensors should have the same length (seq_len)
    seq_len = item["token_ids"].shape[0]
    for key in expected_keys:
        assert item[key].shape == (seq_len,), f"{key} shape mismatch"

    # All should be long tensors
    for key in expected_keys:
        assert item[key].dtype == torch.long, f"{key} dtype mismatch"


def test_dataset_masking_only_keys():
    """Only KEY and LIST_KEY nodes should be masked, never VALUE or LIST_VALUE."""
    vocab = _build_vocab()
    high_mask_config = YamlBertConfig(mask_prob=0.5)
    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
        config=high_mask_config,
    )

    item = dataset[0]
    labels = item["labels"]
    node_types = item["node_types"]

    # Where labels != -100, the node must be KEY (0) or LIST_KEY (2)
    masked_positions = labels != -100
    if masked_positions.any():
        masked_types = node_types[masked_positions]
        for t in masked_types:
            assert t.item() in (0, 2), f"Masked a non-key node: type={t.item()}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement YamlDataset**

File: `yaml_bert/dataset.py`

```python
from __future__ import annotations

import glob
import os
import random

import torch
from torch.utils.data import Dataset

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.config import YamlBertConfig
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.types import NodeType, YamlNode
from yaml_bert.vocab import Vocabulary


# NodeType to integer index mapping
_NODE_TYPE_INDEX: dict[NodeType, int] = {
    NodeType.KEY: 0,
    NodeType.VALUE: 1,
    NodeType.LIST_KEY: 2,
    NodeType.LIST_VALUE: 3,
}

# Node types eligible for masking
_MASKABLE_TYPES: set[NodeType] = {NodeType.KEY, NodeType.LIST_KEY}


class YamlDataset(Dataset):
    """Dataset of linearized, masked YAML documents for YAML-BERT training."""

    def __init__(
        self,
        yaml_dir: str,
        vocab: Vocabulary,
        linearizer: YamlLinearizer,
        annotator: DomainAnnotator,
        config: YamlBertConfig | None = None,
    ) -> None:
        config = config or YamlBertConfig()
        self.vocab: Vocabulary = vocab
        self.linearizer: YamlLinearizer = linearizer
        self.annotator: DomainAnnotator = annotator
        self.mask_prob: float = config.mask_prob
        self.max_seq_len: int = config.max_seq_len

        # Load and linearize all YAML files
        yaml_files: list[str] = sorted(
            glob.glob(os.path.join(yaml_dir, "**", "*.yaml"), recursive=True)
        )
        self.documents: list[list[YamlNode]] = []
        for path in yaml_files:
            nodes: list[YamlNode] = linearizer.linearize_file(path)
            if nodes:
                annotator.annotate(nodes)
                self.documents.append(nodes)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        nodes: list[YamlNode] = self.documents[idx]

        # Truncate if needed
        if len(nodes) > self.max_seq_len:
            nodes = nodes[: self.max_seq_len]

        seq_len: int = len(nodes)

        # Encode nodes to integer arrays
        token_ids: list[int] = []
        node_types: list[int] = []
        depths: list[int] = []
        sibling_indices: list[int] = []
        parent_key_ids: list[int] = []

        for node in nodes:
            # Token ID — route to correct vocab
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                token_ids.append(self.vocab.encode_key(node.token))
            else:
                token_ids.append(self.vocab.encode_value(node.token))

            node_types.append(_NODE_TYPE_INDEX[node.node_type])
            depths.append(min(node.depth, 15))  # clamp to max_depth - 1
            sibling_indices.append(min(node.sibling_index, 31))  # clamp to max_sibling - 1

            parent_key: str = Vocabulary.extract_parent_key(node.parent_path)
            parent_key_ids.append(self.vocab.encode_key(parent_key))

        # Apply masking (only to KEY and LIST_KEY nodes)
        labels: list[int] = [-100] * seq_len
        mask_token_id: int = self.vocab.special_tokens["[MASK]"]

        for i in range(seq_len):
            if nodes[i].node_type not in _MASKABLE_TYPES:
                continue
            if random.random() >= self.mask_prob:
                continue

            # This position is selected for masking
            labels[i] = token_ids[i]  # save original token ID as label

            rand: float = random.random()
            if rand < 0.8:
                # 80%: replace with [MASK]
                token_ids[i] = mask_token_id
            elif rand < 0.9:
                # 10%: replace with random key
                random_key_id: int = random.randint(
                    len(self.vocab.special_tokens),
                    len(self.vocab.key_vocab) + len(self.vocab.special_tokens) - 1,
                )
                token_ids[i] = random_key_id
            # else 10%: keep unchanged

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "node_types": torch.tensor(node_types, dtype=torch.long),
            "depths": torch.tensor(depths, dtype=torch.long),
            "sibling_indices": torch.tensor(sibling_indices, dtype=torch.long),
            "parent_key_ids": torch.tensor(parent_key_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/dataset.py tests/test_dataset.py
git commit -m "feat: YamlDataset with key-only masking"
```

---

### Task 7: Collate Function for Batching

Pad variable-length sequences and create padding masks.

**Files:**
- Modify: `yaml_bert/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_dataset.py`:

```python
from yaml_bert.dataset import collate_fn


def test_collate_fn_padding():
    """Collate should pad to longest sequence in batch and create padding mask."""
    # Simulate two items of different lengths
    item1 = {
        "token_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "node_types": torch.tensor([0, 1, 0], dtype=torch.long),
        "depths": torch.tensor([0, 0, 1], dtype=torch.long),
        "sibling_indices": torch.tensor([0, 0, 0], dtype=torch.long),
        "parent_key_ids": torch.tensor([1, 1, 2], dtype=torch.long),
        "labels": torch.tensor([-100, 5, -100], dtype=torch.long),
    }
    item2 = {
        "token_ids": torch.tensor([4, 5], dtype=torch.long),
        "node_types": torch.tensor([0, 1], dtype=torch.long),
        "depths": torch.tensor([0, 0], dtype=torch.long),
        "sibling_indices": torch.tensor([0, 0], dtype=torch.long),
        "parent_key_ids": torch.tensor([1, 1], dtype=torch.long),
        "labels": torch.tensor([6, -100], dtype=torch.long),
    }

    batch = collate_fn([item1, item2])

    # Should be padded to length 3
    assert batch["token_ids"].shape == (2, 3)
    assert batch["padding_mask"].shape == (2, 3)

    # item2 should be padded in the last position
    assert batch["padding_mask"][0].tolist() == [False, False, False]
    assert batch["padding_mask"][1].tolist() == [False, False, True]

    # Padded positions should have token_id = 0 ([PAD])
    assert batch["token_ids"][1, 2].item() == 0

    # Labels at padded positions should be -100
    assert batch["labels"][1, 2].item() == -100
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_dataset.py::test_collate_fn_padding -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement collate_fn**

Add to `yaml_bert/dataset.py`:

```python
def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences and create padding mask."""
    max_len: int = max(item["token_ids"].size(0) for item in batch)

    padded: dict[str, list[torch.Tensor]] = {key: [] for key in batch[0].keys()}
    padding_masks: list[torch.Tensor] = []

    for item in batch:
        seq_len: int = item["token_ids"].size(0)
        pad_len: int = max_len - seq_len

        for key in item:
            if pad_len > 0:
                pad_value: int = -100 if key == "labels" else 0
                padding: torch.Tensor = torch.full(
                    (pad_len,), pad_value, dtype=torch.long
                )
                padded[key].append(torch.cat([item[key], padding]))
            else:
                padded[key].append(item[key])

        mask: torch.Tensor = torch.cat([
            torch.zeros(seq_len, dtype=torch.bool),
            torch.ones(pad_len, dtype=torch.bool),
        ]) if pad_len > 0 else torch.zeros(seq_len, dtype=torch.bool)
        padding_masks.append(mask)

    result: dict[str, torch.Tensor] = {
        key: torch.stack(tensors) for key, tensors in padded.items()
    }
    result["padding_mask"] = torch.stack(padding_masks)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_dataset.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/dataset.py tests/test_dataset.py
git commit -m "feat: add collate_fn for batch padding"
```

---

### Task 8: YamlBertTrainer

Training loop with optimizer, logging, and checkpointing.

**Files:**
- Create: `yaml_bert/trainer.py`
- Create: `tests/test_trainer.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_trainer.py`

```python
import glob
import os

import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.dataset import YamlDataset
from yaml_bert.trainer import YamlBertTrainer
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "k8s-yamls"
)

TEST_CONFIG: YamlBertConfig = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)


def _build_model_and_dataset() -> tuple[YamlBertModel, YamlDataset]:
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    all_nodes = []
    for path in glob.glob(os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True):
        nodes = linearizer.linearize_file(path)
        annotator.annotate(nodes)
        all_nodes.extend(nodes)

    vocab = VocabBuilder().build(all_nodes)

    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
        config=TEST_CONFIG,
    )

    emb = YamlBertEmbedding(
        config=TEST_CONFIG,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModel(
        config=TEST_CONFIG,
        embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
    )

    return model, dataset


def test_trainer_runs_one_epoch():
    model, dataset = _build_model_and_dataset()

    trainer = YamlBertTrainer(
        config=TEST_CONFIG,
        model=model,
        dataset=dataset,
    )

    losses = trainer.train()

    assert len(losses) == 1
    assert losses[0] > 0


def test_trainer_loss_decreases():
    config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2, num_epochs=5)
    model, dataset = _build_model_and_dataset()

    trainer = YamlBertTrainer(
        config=config,
        model=model,
        dataset=dataset,
    )

    losses = trainer.train()

    assert len(losses) == 5
    # Loss should generally decrease (first epoch > last epoch)
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_trainer_saves_checkpoint(tmp_path):
    model, dataset = _build_model_and_dataset()

    trainer = YamlBertTrainer(
        config=TEST_CONFIG,
        model=model,
        dataset=dataset,
        checkpoint_dir=str(tmp_path),
    )

    trainer.train()

    checkpoint_files = os.listdir(tmp_path)
    assert len(checkpoint_files) > 0
    assert any(f.endswith(".pt") for f in checkpoint_files)

    # Checkpoint should contain model + optimizer state + epoch
    checkpoint_path = os.path.join(tmp_path, checkpoint_files[0])
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "epoch" in checkpoint


def test_trainer_resumes_from_checkpoint(tmp_path):
    model, dataset = _build_model_and_dataset()

    # Train 1 epoch and save checkpoint
    config_1 = YamlBertConfig(d_model=64, num_layers=2, num_heads=2, num_epochs=1)
    trainer1 = YamlBertTrainer(
        config=config_1,
        model=model,
        dataset=dataset,
        checkpoint_dir=str(tmp_path),
        checkpoint_every=1,
    )
    losses1 = trainer1.train()

    # Resume and train 1 more epoch
    checkpoint_path = os.path.join(tmp_path, "yaml_bert_epoch_1.pt")
    config_2 = YamlBertConfig(d_model=64, num_layers=2, num_heads=2, num_epochs=2)
    trainer2 = YamlBertTrainer(
        config=config_2,
        model=model,
        dataset=dataset,
        resume_from=checkpoint_path,
    )
    losses2 = trainer2.train()

    # Should have trained only 1 additional epoch (epoch 2), not from scratch
    assert len(losses2) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trainer.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement YamlBertTrainer**

File: `yaml_bert/trainer.py`

```python
from __future__ import annotations

import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from yaml_bert.config import YamlBertConfig
from yaml_bert.dataset import YamlDataset, collate_fn
from yaml_bert.model import YamlBertModel


class YamlBertTrainer:
    """Training loop for YAML-BERT masked key prediction."""

    def __init__(
        self,
        config: YamlBertConfig,
        model: YamlBertModel,
        dataset: YamlDataset,
        checkpoint_dir: str | None = None,
        checkpoint_every: int = 10,
        resume_from: str | None = None,
    ) -> None:
        self.config: YamlBertConfig = config
        self.model: YamlBertModel = model
        self.dataset: YamlDataset = dataset
        self.checkpoint_dir: str | None = checkpoint_dir
        self.checkpoint_every: int = checkpoint_every
        self.resume_from: str | None = resume_from

        self.device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def train(self) -> list[float]:
        """Run training loop. Returns list of average loss per epoch."""
        self.model.to(self.device)
        self.model.train()

        optimizer: AdamW = AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=0.01
        )

        start_epoch: int = 0

        # Resume from checkpoint if specified
        if self.resume_from:
            checkpoint: dict = torch.load(self.resume_from, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"]
            print(f"Resumed from epoch {start_epoch}")

        dataloader: DataLoader = DataLoader(
            self.dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        epoch_losses: list[float] = []

        for epoch in range(start_epoch, self.config.num_epochs):
            total_loss: float = 0.0
            num_batches: int = 0

            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                optimizer.zero_grad()

                key_logits: torch.Tensor = self.model(
                    token_ids=batch["token_ids"],
                    node_types=batch["node_types"],
                    depths=batch["depths"],
                    sibling_indices=batch["sibling_indices"],
                    parent_key_ids=batch["parent_key_ids"],
                    padding_mask=batch["padding_mask"],
                )

                loss: torch.Tensor = self.model.compute_loss(
                    key_logits, batch["labels"]
                )

                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss: float = total_loss / max(num_batches, 1)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{self.config.num_epochs} — loss: {avg_loss:.4f}")

            # Checkpoint
            if self.checkpoint_dir and (epoch + 1) % self.checkpoint_every == 0:
                self._save_checkpoint(epoch + 1, optimizer)

        # Save final checkpoint
        if self.checkpoint_dir:
            self._save_checkpoint(self.config.num_epochs, optimizer)

        return epoch_losses

    def _save_checkpoint(self, epoch: int, optimizer: AdamW) -> None:
        assert self.checkpoint_dir is not None
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        path: str = os.path.join(
            self.checkpoint_dir, f"yaml_bert_epoch_{epoch}.pt"
        )
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }, path)
        print(f"Checkpoint saved: {path}")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trainer.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/trainer.py tests/test_trainer.py
git commit -m "feat: YamlBertTrainer with training loop and checkpointing"
```

---

### Task 9: Update Package Exports

**Files:**
- Modify: `yaml_bert/__init__.py`

- [ ] **Step 1: Update __init__.py**

File: `yaml_bert/__init__.py`

```python
"""YAML-BERT: Attention on Kubernetes Structured Data."""

from yaml_bert.config import YamlBertConfig
from yaml_bert.types import NodeType, YamlNode
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import Vocabulary, VocabBuilder
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.dataset import YamlDataset, collate_fn
from yaml_bert.trainer import YamlBertTrainer
from yaml_bert.evaluate import YamlBertEvaluator

__all__ = [
    "YamlBertConfig",
    "NodeType",
    "YamlNode",
    "YamlLinearizer",
    "Vocabulary",
    "VocabBuilder",
    "DomainAnnotator",
    "YamlBertEmbedding",
    "YamlBertModel",
    "YamlDataset",
    "collate_fn",
    "YamlBertTrainer",
    "YamlBertEvaluator",
]
```

- [ ] **Step 2: Verify imports**

Run: `python -c "from yaml_bert import YamlBertEmbedding, YamlBertModel, YamlDataset, YamlBertTrainer; print('All Phase 2 exports OK')"`
Expected: `All Phase 2 exports OK`

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add yaml_bert/__init__.py
git commit -m "feat: update package exports with Phase 2 modules"
```

---

### Task 10: End-to-End Integration Test

Full pipeline: load YAML corpus, build vocab, create dataset, build model, train for a few epochs, verify convergence.

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Write end-to-end test**

File: `tests/test_e2e.py`

```python
import glob
import os

import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import YamlDataset
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.trainer import YamlBertTrainer
from yaml_bert.vocab import VocabBuilder

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "k8s-yamls"
)

TEST_CONFIG: YamlBertConfig = YamlBertConfig(
    d_model=64, num_layers=2, num_heads=2, num_epochs=10, batch_size=8,
)


def test_end_to_end_pipeline():
    """Full pipeline: corpus -> vocab -> dataset -> model -> train -> converges."""
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    # Build vocab from corpus
    all_nodes = []
    for path in glob.glob(os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True):
        nodes = linearizer.linearize_file(path)
        annotator.annotate(nodes)
        all_nodes.extend(nodes)

    vocab = VocabBuilder().build(all_nodes)
    print(f"Key vocab: {len(vocab.key_vocab)} tokens")
    print(f"Value vocab: {len(vocab.value_vocab)} tokens")

    # Create dataset
    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
        config=TEST_CONFIG,
    )
    print(f"Dataset: {len(dataset)} documents")

    # Build model
    emb = YamlBertEmbedding(
        config=TEST_CONFIG,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModel(
        config=TEST_CONFIG,
        embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} parameters")

    # Train
    trainer = YamlBertTrainer(
        config=TEST_CONFIG,
        model=model,
        dataset=dataset,
    )
    losses = trainer.train()

    # Verify convergence
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )
    print(f"Loss: {losses[0]:.4f} -> {losses[-1]:.4f}")


def test_tree_position_differentiation():
    """Verify that 'spec' at different tree positions gets different embeddings."""
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    all_nodes = []
    for path in glob.glob(os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True):
        nodes = linearizer.linearize_file(path)
        annotator.annotate(nodes)
        all_nodes.extend(nodes)

    vocab = VocabBuilder().build(all_nodes)

    emb = YamlBertEmbedding(
        config=TEST_CONFIG,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )

    # "spec" at depth 0, parent="root" (root-level spec)
    spec_id = vocab.encode_key("spec")
    token_ids = torch.tensor([[spec_id, spec_id]])
    node_types = torch.tensor([[0, 0]])  # both KEY
    depths = torch.tensor([[0, 2]])  # different depths
    siblings = torch.tensor([[0, 0]])

    # Different parent keys
    root_parent = vocab.encode_key("")  # [UNK] for root
    template_parent = vocab.encode_key("template")
    parent_keys = torch.tensor([[root_parent, template_parent]])

    output = emb(token_ids, node_types, depths, siblings, parent_keys)

    # Same token, different tree positions -> different embeddings
    spec_at_depth0 = output[0, 0]
    spec_at_depth2 = output[0, 1]

    cosine_sim = torch.nn.functional.cosine_similarity(
        spec_at_depth0.unsqueeze(0), spec_at_depth2.unsqueeze(0)
    ).item()

    # Should NOT be identical (cosine sim < 1.0)
    assert cosine_sim < 0.99, (
        f"spec at different depths too similar: cosine_sim={cosine_sim:.4f}"
    )
    print(f"spec at depth 0 vs depth 2: cosine_similarity={cosine_sim:.4f}")
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_e2e.py -v -s`
Expected: ALL PASS (with training output printed)

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test: end-to-end pipeline and tree position differentiation tests"
```

---

### Task 11: Model Evaluation

Post-training evaluation: masked key prediction accuracy, embedding analysis, top-k predictions.

**Files:**
- Create: `yaml_bert/evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_evaluate.py`

```python
import glob
import os

import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.dataset import YamlDataset
from yaml_bert.evaluate import YamlBertEvaluator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "k8s-yamls"
)

TEST_CONFIG: YamlBertConfig = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)


def _build_trained_model() -> tuple[YamlBertModel, YamlDataset, "Vocabulary"]:
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    all_nodes = []
    for path in glob.glob(os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True):
        nodes = linearizer.linearize_file(path)
        annotator.annotate(nodes)
        all_nodes.extend(nodes)

    vocab = VocabBuilder().build(all_nodes)

    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
        config=TEST_CONFIG,
    )

    emb = YamlBertEmbedding(
        config=TEST_CONFIG,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModel(
        config=TEST_CONFIG,
        embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
    )

    # Quick train so model has non-random weights
    from yaml_bert.trainer import YamlBertTrainer
    train_config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2, num_epochs=3)
    trainer = YamlBertTrainer(config=train_config, model=model, dataset=dataset)
    trainer.train()

    return model, dataset, vocab


def test_evaluator_prediction_accuracy():
    model, dataset, vocab = _build_trained_model()
    evaluator = YamlBertEvaluator(model=model, dataset=dataset, vocab=vocab)

    results = evaluator.evaluate_prediction_accuracy()

    assert "top1_accuracy" in results
    assert "top5_accuracy" in results
    assert 0.0 <= results["top1_accuracy"] <= 1.0
    assert 0.0 <= results["top5_accuracy"] <= 1.0
    assert results["top5_accuracy"] >= results["top1_accuracy"]


def test_evaluator_embedding_analysis():
    model, dataset, vocab = _build_trained_model()
    evaluator = YamlBertEvaluator(model=model, dataset=dataset, vocab=vocab)

    results = evaluator.analyze_embeddings()

    # Should report cosine similarities for key pairs
    assert len(results) > 0
    for entry in results:
        assert "key" in entry
        assert "position_a" in entry
        assert "position_b" in entry
        assert "cosine_similarity" in entry
        assert -1.0 <= entry["cosine_similarity"] <= 1.0


def test_evaluator_top_k_predictions():
    model, dataset, vocab = _build_trained_model()
    evaluator = YamlBertEvaluator(model=model, dataset=dataset, vocab=vocab)

    predictions = evaluator.top_k_predictions(doc_idx=0, k=5)

    assert len(predictions) > 0
    for pred in predictions:
        assert "position" in pred
        assert "true_key" in pred
        assert "predicted_keys" in pred
        assert len(pred["predicted_keys"]) <= 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_evaluate.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement YamlBertEvaluator**

File: `yaml_bert/evaluate.py`

```python
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from yaml_bert.dataset import YamlDataset
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary


class YamlBertEvaluator:
    """Post-training evaluation for YAML-BERT."""

    def __init__(
        self,
        model: YamlBertModel,
        dataset: YamlDataset,
        vocab: Vocabulary,
    ) -> None:
        self.model: YamlBertModel = model
        self.dataset: YamlDataset = dataset
        self.vocab: Vocabulary = vocab
        self.device: torch.device = next(model.parameters()).device

    @torch.no_grad()
    def evaluate_prediction_accuracy(self) -> dict[str, float]:
        """Compute top-1 and top-5 masked key prediction accuracy over the dataset."""
        self.model.eval()

        total_masked: int = 0
        top1_correct: int = 0
        top5_correct: int = 0

        for idx in range(len(self.dataset)):
            item: dict[str, torch.Tensor] = self.dataset[idx]
            labels: torch.Tensor = item["labels"]

            masked_positions: torch.Tensor = labels != -100
            if not masked_positions.any():
                continue

            # Add batch dimension
            batch: dict[str, torch.Tensor] = {
                k: v.unsqueeze(0).to(self.device) for k, v in item.items()
            }

            key_logits: torch.Tensor = self.model(
                token_ids=batch["token_ids"],
                node_types=batch["node_types"],
                depths=batch["depths"],
                sibling_indices=batch["sibling_indices"],
                parent_key_ids=batch["parent_key_ids"],
            )

            logits: torch.Tensor = key_logits[0]  # remove batch dim
            for pos in masked_positions.nonzero(as_tuple=True)[0]:
                true_id: int = labels[pos].item()
                pos_logits: torch.Tensor = logits[pos]
                top5_ids: torch.Tensor = pos_logits.topk(5).indices

                if top5_ids[0].item() == true_id:
                    top1_correct += 1
                if true_id in top5_ids.tolist():
                    top5_correct += 1
                total_masked += 1

        return {
            "top1_accuracy": top1_correct / max(total_masked, 1),
            "top5_accuracy": top5_correct / max(total_masked, 1),
            "total_masked": total_masked,
        }

    @torch.no_grad()
    def analyze_embeddings(self) -> list[dict[str, Any]]:
        """Compare embeddings of the same key at different tree positions."""
        self.model.eval()
        results: list[dict[str, Any]] = []

        # Test pairs: same key, different (depth, parent_key) combinations
        test_pairs: list[dict[str, Any]] = [
            {
                "key": "spec",
                "position_a": {"depth": 0, "parent_key": ""},
                "position_b": {"depth": 2, "parent_key": "template"},
            },
            {
                "key": "name",
                "position_a": {"depth": 1, "parent_key": "metadata"},
                "position_b": {"depth": 1, "parent_key": "containers"},
            },
        ]

        for pair in test_pairs:
            key_id: int = self.vocab.encode_key(pair["key"])

            token_ids: torch.Tensor = torch.tensor(
                [[key_id, key_id]], device=self.device
            )
            node_types: torch.Tensor = torch.tensor(
                [[0, 0]], device=self.device
            )
            depths: torch.Tensor = torch.tensor(
                [[pair["position_a"]["depth"], pair["position_b"]["depth"]]],
                device=self.device,
            )
            siblings: torch.Tensor = torch.tensor(
                [[0, 0]], device=self.device
            )
            parent_a_id: int = self.vocab.encode_key(
                pair["position_a"]["parent_key"]
            )
            parent_b_id: int = self.vocab.encode_key(
                pair["position_b"]["parent_key"]
            )
            parent_keys: torch.Tensor = torch.tensor(
                [[parent_a_id, parent_b_id]], device=self.device
            )

            embeddings: torch.Tensor = self.model.embedding(
                token_ids, node_types, depths, siblings, parent_keys
            )

            cosine_sim: float = F.cosine_similarity(
                embeddings[0, 0].unsqueeze(0),
                embeddings[0, 1].unsqueeze(0),
            ).item()

            results.append({
                "key": pair["key"],
                "position_a": pair["position_a"],
                "position_b": pair["position_b"],
                "cosine_similarity": cosine_sim,
            })

        return results

    @torch.no_grad()
    def top_k_predictions(
        self, doc_idx: int, k: int = 5
    ) -> list[dict[str, Any]]:
        """Show top-k predicted keys for each masked position in a document."""
        self.model.eval()

        item: dict[str, torch.Tensor] = self.dataset[doc_idx]
        labels: torch.Tensor = item["labels"]

        masked_positions: torch.Tensor = labels != -100
        if not masked_positions.any():
            return []

        batch: dict[str, torch.Tensor] = {
            k_: v.unsqueeze(0).to(self.device) for k_, v in item.items()
        }

        key_logits: torch.Tensor = self.model(
            token_ids=batch["token_ids"],
            node_types=batch["node_types"],
            depths=batch["depths"],
            sibling_indices=batch["sibling_indices"],
            parent_key_ids=batch["parent_key_ids"],
        )

        logits: torch.Tensor = key_logits[0]
        predictions: list[dict[str, Any]] = []

        for pos in masked_positions.nonzero(as_tuple=True)[0]:
            true_id: int = labels[pos].item()
            pos_logits: torch.Tensor = logits[pos]
            probs: torch.Tensor = F.softmax(pos_logits, dim=-1)
            topk: torch.return_types.topk = probs.topk(k)

            predicted_keys: list[dict[str, Any]] = [
                {
                    "key": self.vocab.decode_key(topk.indices[i].item()),
                    "probability": topk.values[i].item(),
                }
                for i in range(k)
            ]

            predictions.append({
                "position": pos.item(),
                "true_key": self.vocab.decode_key(true_id),
                "predicted_keys": predicted_keys,
            })

        return predictions
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_evaluate.py -v -s`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/evaluate.py tests/test_evaluate.py
git commit -m "feat: YamlBertEvaluator with accuracy, embedding analysis, top-k predictions"
```

---

### Task 12: Training Visualizations

Generate plots for training loss curve, embedding similarity heatmap, and attention patterns. Saves as PNG files.

**Files:**
- Create: `yaml_bert/visualize.py`
- Create: `tests/test_visualize.py`

- [ ] **Step 1: Add matplotlib dependency**

Add to `requirements.txt`:

```
pyyaml>=6.0
pytest>=7.0
torch>=2.0
matplotlib>=3.7
```

Run: `pip install -r requirements.txt`

- [ ] **Step 2: Write failing test**

File: `tests/test_visualize.py`

```python
import os

from yaml_bert.visualize import plot_training_loss, plot_embedding_similarity, plot_attention_patterns


def test_plot_training_loss(tmp_path):
    losses = [5.2, 4.8, 4.1, 3.5, 3.0, 2.7, 2.4, 2.2, 2.0, 1.9]
    output_path = str(tmp_path / "loss.png")

    plot_training_loss(losses, output_path=output_path)

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_plot_embedding_similarity(tmp_path):
    embedding_results = [
        {
            "key": "spec",
            "position_a": {"depth": 0, "parent_key": ""},
            "position_b": {"depth": 2, "parent_key": "template"},
            "cosine_similarity": 0.45,
        },
        {
            "key": "name",
            "position_a": {"depth": 1, "parent_key": "metadata"},
            "position_b": {"depth": 1, "parent_key": "containers"},
            "cosine_similarity": 0.32,
        },
    ]
    output_path = str(tmp_path / "embeddings.png")

    plot_embedding_similarity(embedding_results, output_path=output_path)

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_plot_attention_patterns(tmp_path):
    import torch
    # Fake attention weights: (num_heads, seq_len, seq_len)
    attention_weights = torch.rand(2, 8, 8)
    token_labels = ["apiVersion", "apps/v1", "kind", "Deployment",
                    "metadata", "name", "nginx", "spec"]
    output_path = str(tmp_path / "attention.png")

    plot_attention_patterns(
        attention_weights,
        token_labels=token_labels,
        output_path=output_path,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_visualize.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 4: Implement visualization functions**

File: `yaml_bert/visualize.py`

```python
from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import torch


def plot_training_loss(
    losses: list[float],
    output_path: str = "training_loss.png",
    title: str = "YAML-BERT Training Loss",
) -> None:
    """Plot training loss curve over epochs."""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(losses) + 1), losses, marker="o", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Training loss plot saved: {output_path}")


def plot_embedding_similarity(
    results: list[dict[str, Any]],
    output_path: str = "embedding_similarity.png",
    title: str = "Tree Position Embedding Similarity",
) -> None:
    """Plot cosine similarity between same-key embeddings at different tree positions."""
    labels: list[str] = []
    similarities: list[float] = []

    for r in results:
        pa: dict[str, Any] = r["position_a"]
        pb: dict[str, Any] = r["position_b"]
        label: str = (
            f"{r['key']}\n"
            f"d={pa['depth']},p={pa['parent_key']}\n"
            f"vs d={pb['depth']},p={pb['parent_key']}"
        )
        labels.append(label)
        similarities.append(r["cosine_similarity"])

    fig, ax = plt.subplots(figsize=(max(8, len(results) * 3), 6))
    bars = ax.bar(range(len(results)), similarities, color="steelblue")
    ax.set_xticks(range(len(results)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(title)
    ax.set_ylim(-1, 1)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    for bar, sim in zip(bars, similarities):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{sim:.3f}",
            ha="center",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Embedding similarity plot saved: {output_path}")


def plot_attention_patterns(
    attention_weights: torch.Tensor,
    token_labels: list[str],
    output_path: str = "attention_patterns.png",
    title: str = "Attention Patterns",
) -> None:
    """Plot attention heatmaps for each head.

    Args:
        attention_weights: (num_heads, seq_len, seq_len)
        token_labels: labels for each position in the sequence
    """
    num_heads: int = attention_weights.shape[0]
    fig, axes = plt.subplots(1, num_heads, figsize=(6 * num_heads, 5))

    if num_heads == 1:
        axes = [axes]

    for head_idx, ax in enumerate(axes):
        weights: torch.Tensor = attention_weights[head_idx].cpu()
        im = ax.imshow(weights.numpy(), cmap="Blues", vmin=0, vmax=1)
        ax.set_title(f"Head {head_idx}")
        ax.set_xticks(range(len(token_labels)))
        ax.set_yticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(token_labels, fontsize=7)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Attention pattern plot saved: {output_path}")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_visualize.py -v`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add yaml_bert/visualize.py tests/test_visualize.py requirements.txt
git commit -m "feat: visualization functions for loss, embeddings, and attention"
```
