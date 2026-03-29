# Kind Embedding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `kind_embedding` as a document-level positional component so the model distinguishes kind-specific structures (e.g., Deployment spec vs Pod spec).

**Architecture:** Extract the `kind` value from each document during dataset creation, build a small kind vocabulary (~50 entries), add an optional `kind_embedding` table to the embedding layer. Every node in a document gets the same kind vector added to its positional encoding. Backward compatible — v1 checkpoints load without kind_embedding.

**Tech Stack:** Python 3.10+, PyTorch >= 2.0

**Spec:** `docs/superpowers/specs/2026-03-29-kind-embedding-design.md`

---

### Task 1: Kind Vocabulary in VocabBuilder

Add kind vocabulary building to `Vocabulary` and `VocabBuilder`. Kind vocab is separate from key/value vocabs — a small dedicated vocabulary of resource type names.

**Files:**
- Modify: `yaml_bert/vocab.py`
- Create: `tests/test_kind_vocab.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_kind_vocab.py`

```python
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import Vocabulary, VocabBuilder


def test_build_kind_vocab():
    linearizer = YamlLinearizer()
    nodes1 = linearizer.linearize("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
""")
    nodes2 = linearizer.linearize("""\
apiVersion: v1
kind: Service
metadata:
  name: test
""")
    nodes3 = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
""")

    builder = VocabBuilder()
    vocab = builder.build(nodes1 + nodes2 + nodes3)

    assert vocab.kind_vocab is not None
    assert "Deployment" in vocab.kind_vocab
    assert "Service" in vocab.kind_vocab
    assert "Pod" in vocab.kind_vocab


def test_encode_kind():
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
""")
    vocab = VocabBuilder().build(nodes)

    kind_id = vocab.encode_kind("Pod")
    assert kind_id != vocab.kind_vocab.get("[NO_KIND]", -1)
    assert kind_id == vocab.kind_vocab["Pod"]

    # Unknown kind
    unknown_id = vocab.encode_kind("UnknownResource")
    assert unknown_id == vocab.kind_vocab["[NO_KIND]"]


def test_no_kind_document():
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("""\
apiVersion: v1
data:
  key: value
""")
    vocab = VocabBuilder().build(nodes)

    no_kind_id = vocab.encode_kind("")
    assert no_kind_id == vocab.kind_vocab["[NO_KIND]"]


def test_kind_vocab_saved_and_loaded(tmp_path):
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
""")
    vocab = VocabBuilder().build(nodes)

    path = str(tmp_path / "vocab.json")
    vocab.save(path)
    loaded = Vocabulary.load(path)

    assert loaded.kind_vocab == vocab.kind_vocab
    assert loaded.encode_kind("Pod") == vocab.encode_kind("Pod")


def test_kind_vocab_size():
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
""")
    vocab = VocabBuilder().build(nodes)

    # kind_vocab_size includes [NO_KIND] + actual kinds
    assert vocab.kind_vocab_size >= 2  # at least [NO_KIND] + Pod


def test_backward_compatible_load(tmp_path):
    """Loading a v1 vocab (no kind_vocab) should work — kind_vocab defaults to empty."""
    import json
    v1_data = {
        "key_vocab": {"apiVersion": 3, "kind": 4},
        "value_vocab": {"v1": 3, "Pod": 4},
        "special_tokens": {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2},
    }
    path = str(tmp_path / "v1_vocab.json")
    with open(path, "w") as f:
        json.dump(v1_data, f)

    loaded = Vocabulary.load(path)
    assert loaded.kind_vocab == {"[NO_KIND]": 0}
    assert loaded.encode_kind("Pod") == 0  # maps to [NO_KIND]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_kind_vocab.py -v`
Expected: FAIL

- [ ] **Step 3: Implement kind vocabulary**

Modify `yaml_bert/vocab.py`:

Add `kind_vocab` to `Vocabulary.__init__`:

```python
class Vocabulary:
    def __init__(
        self,
        key_vocab: dict[str, int],
        value_vocab: dict[str, int],
        special_tokens: dict[str, int],
        kind_vocab: dict[str, int] | None = None,
    ) -> None:
        self.key_vocab = key_vocab
        self.value_vocab = value_vocab
        self.special_tokens = special_tokens
        self.kind_vocab = kind_vocab or {"[NO_KIND]": 0}
        self._id_to_key = {v: k for k, v in key_vocab.items()}
        self._id_to_value = {v: k for k, v in value_vocab.items()}
        self._id_to_special = {v: k for k, v in special_tokens.items()}
```

Add `encode_kind`, `kind_vocab_size`:

```python
    def encode_kind(self, kind: str) -> int:
        if not kind:
            return self.kind_vocab["[NO_KIND]"]
        return self.kind_vocab.get(kind, self.kind_vocab["[NO_KIND]"])

    @property
    def kind_vocab_size(self) -> int:
        return len(self.kind_vocab)
```

Update `save` to include `kind_vocab`:

```python
    def save(self, path: str) -> None:
        data = {
            "key_vocab": self.key_vocab,
            "value_vocab": self.value_vocab,
            "special_tokens": self.special_tokens,
            "kind_vocab": self.kind_vocab,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
```

Update `load` for backward compatibility:

```python
    @classmethod
    def load(cls, path: str) -> Vocabulary:
        with open(path) as f:
            data = json.load(f)
        return cls(
            key_vocab=data["key_vocab"],
            value_vocab=data["value_vocab"],
            special_tokens=data["special_tokens"],
            kind_vocab=data.get("kind_vocab"),
        )
```

Update `VocabBuilder.build` to extract kinds:

```python
    def build(self, nodes: list[YamlNode], min_freq: int = 1) -> Vocabulary:
        key_counts: dict[str, int] = {}
        value_counts: dict[str, int] = {}
        kind_set: set[str] = set()

        prev_token: str = ""
        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                key_counts[node.token] = key_counts.get(node.token, 0) + 1
                prev_token = node.token
            elif node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE):
                value_counts[node.token] = value_counts.get(node.token, 0) + 1
                if prev_token == "kind" and node.depth == 0:
                    kind_set.add(node.token)
                prev_token = ""

        kind_vocab: dict[str, int] = {"[NO_KIND]": 0}
        for i, kind in enumerate(sorted(kind_set)):
            kind_vocab[kind] = i + 1

        return Vocabulary(key_vocab, value_vocab, special_tokens, kind_vocab)
```

Note: the `build_from_counts` static method also needs updating — add `kind_counts` parameter. And `build_from_huggingface` needs to collect kinds during scanning.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_kind_vocab.py -v`
Expected: ALL PASS

- [ ] **Step 5: Run all existing tests**

Run: `pytest tests/ -q --ignore=tests/test_trainer.py --ignore=tests/test_e2e.py --ignore=tests/test_evaluate.py`
Expected: ALL PASS

- [ ] **Step 6: Commit**

```bash
git add yaml_bert/vocab.py tests/test_kind_vocab.py
git commit -m "feat: add kind vocabulary to Vocabulary and VocabBuilder"
```

---

### Task 2: Kind Embedding in Embedding Layer

Add optional `kind_embedding` table. When `kind_ids` is passed, add it to the positional encoding. When not passed, skip it. V1 checkpoints load with `strict=False`.

**Files:**
- Modify: `yaml_bert/embedding.py`
- Modify: `tests/test_embedding.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_embedding.py`:

```python
def test_embedding_with_kind():
    config = YamlBertConfig(d_model=32)
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=10,
        value_vocab_size=10,
        kind_vocab_size=5,
    )

    token_ids = torch.tensor([[3, 3]])
    node_types = torch.tensor([[0, 0]])
    depths = torch.tensor([[1, 1]])
    siblings = torch.tensor([[0, 0]])
    parent_keys = torch.tensor([[4, 4]])
    kind_ids = torch.tensor([[1, 1]])  # same kind for all nodes

    output = emb(token_ids, node_types, depths, siblings, parent_keys, kind_ids=kind_ids)
    assert output.shape == (1, 2, 32)


def test_embedding_different_kinds_produce_different_output():
    config = YamlBertConfig(d_model=32)
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=10,
        value_vocab_size=10,
        kind_vocab_size=5,
    )

    token_ids = torch.tensor([[3]])
    node_types = torch.tensor([[0]])
    depths = torch.tensor([[1]])
    siblings = torch.tensor([[0]])
    parent_keys = torch.tensor([[4]])

    kind_a = torch.tensor([[1]])
    kind_b = torch.tensor([[2]])

    out_a = emb(token_ids, node_types, depths, siblings, parent_keys, kind_ids=kind_a)
    out_b = emb(token_ids, node_types, depths, siblings, parent_keys, kind_ids=kind_b)

    assert not torch.allclose(out_a, out_b)


def test_embedding_without_kind_backward_compatible():
    config = YamlBertConfig(d_model=32)
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=10,
        value_vocab_size=10,
    )

    token_ids = torch.tensor([[3]])
    node_types = torch.tensor([[0]])
    depths = torch.tensor([[1]])
    siblings = torch.tensor([[0]])
    parent_keys = torch.tensor([[4]])

    # No kind_ids — should work (backward compatible)
    output = emb(token_ids, node_types, depths, siblings, parent_keys)
    assert output.shape == (1, 1, 32)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_embedding.py -v`
Expected: FAIL (new tests fail, old tests pass)

- [ ] **Step 3: Update YamlBertEmbedding**

Modify `yaml_bert/embedding.py` — add optional `kind_vocab_size` and `kind_ids`:

```python
class YamlBertEmbedding(nn.Module):
    def __init__(
        self,
        config: YamlBertConfig,
        key_vocab_size: int,
        value_vocab_size: int,
        kind_vocab_size: int | None = None,
    ) -> None:
        super().__init__()

        d: int = config.d_model

        self.key_embedding: nn.Embedding = nn.Embedding(key_vocab_size, d)
        self.value_embedding: nn.Embedding = nn.Embedding(value_vocab_size, d)

        self.depth_embedding: nn.Embedding = nn.Embedding(config.max_depth, d)
        self.sibling_embedding: nn.Embedding = nn.Embedding(config.max_sibling, d)
        self.node_type_embedding: nn.Embedding = nn.Embedding(4, d)
        self.parent_key_embedding: nn.Embedding = nn.Embedding(key_vocab_size, d)

        self.kind_embedding: nn.Embedding | None = None
        if kind_vocab_size is not None:
            self.kind_embedding = nn.Embedding(kind_vocab_size, d)

        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d)

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        parent_key_ids: torch.Tensor,
        kind_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        is_key: torch.Tensor = (node_types == 0) | (node_types == 2)

        key_vocab_size: int = self.key_embedding.num_embeddings
        val_vocab_size: int = self.value_embedding.num_embeddings
        key_emb: torch.Tensor = self.key_embedding(token_ids.clamp(0, key_vocab_size - 1))
        val_emb: torch.Tensor = self.value_embedding(token_ids.clamp(0, val_vocab_size - 1))
        token_emb: torch.Tensor = torch.where(
            is_key.unsqueeze(-1), key_emb, val_emb
        )

        tree_pos: torch.Tensor = (
            self.depth_embedding(depths)
            + self.sibling_embedding(sibling_indices)
            + self.node_type_embedding(node_types)
            + self.parent_key_embedding(parent_key_ids)
        )

        if self.kind_embedding is not None and kind_ids is not None:
            tree_pos = tree_pos + self.kind_embedding(kind_ids)

        return self.layer_norm(token_emb + tree_pos)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_embedding.py -v`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/embedding.py tests/test_embedding.py
git commit -m "feat: add optional kind_embedding to YamlBertEmbedding"
```

---

### Task 3: Pass kind_ids Through Model

Update `YamlBertModel.forward` and `get_attention_weights` to accept and pass `kind_ids`.

**Files:**
- Modify: `yaml_bert/model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_model.py`:

```python
def test_model_with_kind_ids():
    config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=100,
        value_vocab_size=200,
        kind_vocab_size=10,
    )
    model = YamlBertModel(
        config=config,
        embedding=emb,
        key_vocab_size=100,
    )

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))
    kind_ids = torch.tensor([[1, 1, 1, 1, 1], [2, 2, 2, 2, 2]])

    key_logits = model(
        token_ids, node_types, depths, siblings, parent_keys,
        kind_ids=kind_ids,
    )
    assert key_logits.shape == (batch_size, seq_len, 100)


def test_model_without_kind_ids_backward_compatible():
    model = _make_model()
    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))

    # No kind_ids — should work
    key_logits = model(token_ids, node_types, depths, siblings, parent_keys)
    assert key_logits.shape == (batch_size, seq_len, KEY_VOCAB_SIZE)
```

- [ ] **Step 2: Update model.py**

Add `kind_ids: torch.Tensor | None = None` to `forward` and `get_attention_weights`, pass through to `self.embedding`:

```python
    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        parent_key_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        kind_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x: torch.Tensor = self.embedding(
            token_ids, node_types, depths, sibling_indices, parent_key_ids,
            kind_ids=kind_ids,
        )
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        key_logits: torch.Tensor = self.key_prediction_head(x)
        return key_logits
```

Same for `get_attention_weights` — add `kind_ids` parameter and pass it.

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: ALL PASS (new and old tests)

- [ ] **Step 4: Commit**

```bash
git add yaml_bert/model.py tests/test_model.py
git commit -m "feat: pass kind_ids through YamlBertModel"
```

---

### Task 4: Extract kind_ids in YamlDataset

Extract the `kind` value from each document's node list and include `kind_ids` in the output tensors. Update `collate_fn` to pad `kind_ids`.

**Files:**
- Modify: `yaml_bert/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_dataset.py`:

```python
def test_dataset_includes_kind_ids():
    vocab = _build_vocab()
    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
    )
    item = dataset[0]

    assert "kind_ids" in item
    # kind_ids should be same value repeated for all nodes (document-level)
    kind_ids = item["kind_ids"]
    assert kind_ids.shape == item["token_ids"].shape
    assert (kind_ids == kind_ids[0]).all(), "All kind_ids should be the same within a document"


def test_collate_fn_pads_kind_ids():
    item1 = {
        "token_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "node_types": torch.tensor([0, 1, 0], dtype=torch.long),
        "depths": torch.tensor([0, 0, 1], dtype=torch.long),
        "sibling_indices": torch.tensor([0, 0, 0], dtype=torch.long),
        "parent_key_ids": torch.tensor([1, 1, 2], dtype=torch.long),
        "labels": torch.tensor([-100, 5, -100], dtype=torch.long),
        "kind_ids": torch.tensor([3, 3, 3], dtype=torch.long),
    }
    item2 = {
        "token_ids": torch.tensor([4, 5], dtype=torch.long),
        "node_types": torch.tensor([0, 1], dtype=torch.long),
        "depths": torch.tensor([0, 0], dtype=torch.long),
        "sibling_indices": torch.tensor([0, 0], dtype=torch.long),
        "parent_key_ids": torch.tensor([1, 1], dtype=torch.long),
        "labels": torch.tensor([6, -100], dtype=torch.long),
        "kind_ids": torch.tensor([5, 5], dtype=torch.long),
    }

    batch = collate_fn([item1, item2])

    assert "kind_ids" in batch
    assert batch["kind_ids"].shape == (2, 3)
    assert batch["kind_ids"][1, 2].item() == 0  # padded with 0
```

- [ ] **Step 2: Update dataset.py**

Add kind extraction helper:

```python
def _extract_kind(nodes: list[YamlNode]) -> str:
    """Extract the kind value from a document's node list."""
    for i, node in enumerate(nodes):
        if (node.token == "kind"
            and node.depth == 0
            and node.node_type == NodeType.KEY
            and i + 1 < len(nodes)
            and nodes[i + 1].node_type == NodeType.VALUE):
            return nodes[i + 1].token
    return ""
```

In `__init__`, store kind per document:

```python
        self.document_kinds: list[str] = []
        for path in yaml_files:
            nodes: list[YamlNode] = linearizer.linearize_file(path)
            if nodes:
                annotator.annotate(nodes)
                self.documents.append(nodes)
                self.document_kinds.append(_extract_kind(nodes))
```

Same for `from_huggingface`.

In `__getitem__`, add `kind_ids`:

```python
        kind: str = self.document_kinds[idx]
        kind_id: int = self.vocab.encode_kind(kind)
        kind_ids: list[int] = [kind_id] * seq_len
```

Add to return dict:

```python
            "kind_ids": torch.tensor(kind_ids, dtype=torch.long),
```

The `collate_fn` already handles any key generically — `kind_ids` will be padded with 0 automatically.

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_dataset.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add yaml_bert/dataset.py tests/test_dataset.py
git commit -m "feat: extract kind_ids in YamlDataset"
```

---

### Task 5: Pass kind_ids in Trainer and Evaluator

Update trainer and evaluator to pass `kind_ids` from batches to the model.

**Files:**
- Modify: `yaml_bert/trainer.py`
- Modify: `yaml_bert/evaluate.py`

- [ ] **Step 1: Update trainer.py**

In the training loop, add `kind_ids` to the model call:

```python
                key_logits: torch.Tensor = self.model(
                    token_ids=batch["token_ids"],
                    node_types=batch["node_types"],
                    depths=batch["depths"],
                    sibling_indices=batch["sibling_indices"],
                    parent_key_ids=batch["parent_key_ids"],
                    padding_mask=batch["padding_mask"],
                    kind_ids=batch.get("kind_ids"),
                )
```

Using `batch.get("kind_ids")` for backward compatibility — if `kind_ids` isn't in the batch (v1 dataset), it returns `None`.

- [ ] **Step 2: Update evaluate.py**

Same pattern in `evaluate_prediction_accuracy` and `top_k_predictions` — add `kind_ids=batch.get("kind_ids")` to model calls.

- [ ] **Step 3: Run existing tests**

Run: `pytest tests/ -q --ignore=tests/test_trainer.py --ignore=tests/test_e2e.py --ignore=tests/test_evaluate.py`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add yaml_bert/trainer.py yaml_bert/evaluate.py
git commit -m "feat: pass kind_ids through trainer and evaluator"
```

---

### Task 6: Update Package Exports and __init__.py

**Files:**
- Modify: `yaml_bert/__init__.py`

- [ ] **Step 1: No new exports needed**

The `kind_vocab` is part of `Vocabulary`, `kind_embedding` is part of `YamlBertEmbedding`. No new classes to export.

Verify imports still work:

Run: `python -c "from yaml_bert import YamlBertConfig, YamlBertEmbedding, YamlBertModel, YamlDataset, YamlBertTrainer, Vocabulary; print('OK')"`
Expected: `OK`

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -q --ignore=tests/test_trainer.py --ignore=tests/test_e2e.py --ignore=tests/test_evaluate.py`
Expected: ALL PASS

- [ ] **Step 3: Commit**

```bash
git commit --allow-empty -m "chore: verify all exports and tests pass with kind embedding"
```

---

### Task 7: Update Training Script

Update `scripts/train_hf.py` to build kind vocab and pass `kind_vocab_size` to the model.

**Files:**
- Modify: `scripts/train_hf.py`

- [ ] **Step 1: Update model creation in train_hf.py**

Where the model is built, add `kind_vocab_size`:

```python
    emb: YamlBertEmbedding = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
```

The vocab already has `kind_vocab` from the updated `VocabBuilder`. The dataset already extracts `kind_ids`. The trainer already passes them. This is the only change needed.

- [ ] **Step 2: Update export_model.py to include kind info**

Add kind vocab size to the exported metadata.

- [ ] **Step 3: Commit**

```bash
git add scripts/train_hf.py scripts/export_model.py
git commit -m "feat: update training script for kind embedding"
```

---

### Task 8: Update Evaluation and Visualization Scripts

Update all scripts in `scripts/` and `model_tests/` to pass `kind_ids` when using v2 checkpoints.

**Files:**
- Modify: `scripts/evaluate_checkpoint.py`
- Modify: `scripts/anomaly_score.py`
- Modify: `scripts/visualize_attention.py`
- Modify: `scripts/visualize_tree.py`
- Modify: `scripts/visualize_examples.py`
- Modify: `scripts/evaluate_all.py`
- Modify: `model_tests/test_structural.py`
- Modify: `model_tests/test_capabilities.py`

- [ ] **Step 1: Update scripts to detect kind_embedding and pass kind_ids**

The pattern for each script: when building tensors for inference, also compute `kind_id` from the document and create `kind_ids` tensor. Pass to model. Use `vocab.encode_kind()`.

For `anomaly_score.py`, the `score_yaml` function needs to extract kind and pass it:

```python
    # Extract kind from document
    kind: str = _extract_kind(nodes)
    kind_id: int = vocab.encode_kind(kind)
    kind_ids: list[int] = [kind_id] * len(nodes)
```

For `test_capabilities.py`, the `run_test` and `predict_masked` functions need the same update.

- [ ] **Step 2: Verify v1 checkpoint still works**

Run: `python scripts/anomaly_score.py output_v1/yaml_bert_v1_final.pt --run-examples`
Expected: Works (kind_ids not passed because v1 model has no kind_embedding)

- [ ] **Step 3: Commit**

```bash
git add scripts/ model_tests/
git commit -m "feat: update all scripts to support kind_ids"
```

---

### Task 9: Add New Capability Tests

Add the kind-specific capability tests from the spec.

**Files:**
- Modify: `model_tests/test_capabilities.py`

- [ ] **Step 1: Add 4 new capabilities**

Add to `build_capabilities()`:

- Capability 21: Kind-specific spec children (11 test cases)
- Capability 22: Kind-specific invalid structure rejection (12 test cases)
- Capability 23: Same structure different kind (4 test cases)
- Capability 24: Kind embedding does not harm valid structures (5 test cases)

These tests are designed to pass only with kind embedding. They will fail on v1 checkpoints — that's expected and proves the value of kind embedding.

- [ ] **Step 2: Commit**

```bash
git add model_tests/test_capabilities.py
git commit -m "feat: add kind-specific capability tests (32 new test cases)"
```

---

### Task 10: Retrain and Evaluate

Train v2 model with kind embedding on the full 276K dataset.

- [ ] **Step 1: Rebuild vocab with kind support**

```bash
rm output_v1/token_counts.json output_v1/vocab.json
python scripts/train_hf.py --max-docs 0 --full --epochs 15 --vocab-min-freq 100
```

Output goes to `output_v1/` (or configure `--output-dir output_v2/`).

- [ ] **Step 2: Run capability tests on v2 checkpoint**

```bash
python model_tests/test_capabilities.py output_v2/checkpoints/yaml_bert_epoch_10.pt
```

- [ ] **Step 3: Run anomaly detection comparison**

```bash
python scripts/anomaly_score.py output_v1/yaml_bert_v1_final.pt --run-examples
python scripts/anomaly_score.py output_v2/checkpoints/yaml_bert_epoch_10.pt --run-examples
```

Compare detection rates.

- [ ] **Step 4: Export final v2 model**

```bash
python scripts/export_model.py output_v2/checkpoints/yaml_bert_epoch_15.pt --output output_v2/yaml_bert_v2_final.pt
```
