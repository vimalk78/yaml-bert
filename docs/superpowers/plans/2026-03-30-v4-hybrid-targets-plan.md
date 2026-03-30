# v4 Hybrid Targets Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement hybrid bigram/trigram prediction targets — simple targets for universal structure (metadata), kind-specific trigram targets for spec/data/rules children. Two prediction heads, simplified embedding (4 tables instead of 6).

**Architecture:** The embedding layer drops kind_embedding and parent_key_embedding. Two prediction heads: simple (bigrams for universal keys) and kind-specific (trigrams for first-level children of kind-specific root keys). The target type is determined by tree position: `UNIVERSAL_ROOT_KEYS = {apiVersion, kind, metadata}` — children under anything else get trigrams.

**Tech Stack:** Python 3.10+, PyTorch >= 2.0

**Spec:** `docs/superpowers/specs/2026-03-30-v4-architecture-design.md`

---

### Task 1: Build hybrid target vocabularies

Build the simple (bigram) and kind-specific (trigram) target vocabularies from the cached documents.

**Files:**
- Modify: `yaml_bert/vocab.py`
- Create: `tests/test_hybrid_vocab.py`

- [ ] **Step 1: Write failing test**

File: `tests/test_hybrid_vocab.py`

```python
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder, Vocabulary


def test_build_hybrid_vocabs():
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes = linearizer.linearize("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
""")
    annotator.annotate(nodes)
    vocab = VocabBuilder().build(nodes)

    assert hasattr(vocab, "simple_target_vocab")
    assert hasattr(vocab, "kind_target_vocab")

    # Simple targets: unigrams (root keys) + bigrams (metadata children, deeper nodes)
    assert "apiVersion" in vocab.simple_target_vocab       # unigram
    assert "metadata" in vocab.simple_target_vocab         # unigram
    assert "metadata::name" in vocab.simple_target_vocab   # bigram
    assert "metadata::labels" in vocab.simple_target_vocab # bigram
    assert "matchLabels::app" in vocab.simple_target_vocab # bigram (deeper under spec)

    # Kind-specific targets: trigrams for first-level spec children
    assert "Deployment::spec::replicas" in vocab.kind_target_vocab
    assert "Deployment::spec::selector" in vocab.kind_target_vocab


def test_hybrid_vocab_sizes():
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  ports:
  - port: 80
  selector:
    app: web
  type: ClusterIP
""")
    annotator.annotate(nodes)
    vocab = VocabBuilder().build(nodes)

    assert vocab.simple_target_vocab_size > 0
    assert vocab.kind_target_vocab_size > 0


def test_encode_simple_target():
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

    assert vocab.encode_simple_target("metadata::name") != vocab.special_tokens["[UNK]"]
    assert vocab.encode_simple_target("nonexistent::key") == vocab.special_tokens["[UNK]"]


def test_encode_kind_target():
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
""")
    annotator.annotate(nodes)
    vocab = VocabBuilder().build(nodes)

    assert vocab.encode_kind_target("Pod::spec::containers") != vocab.special_tokens["[UNK]"]


def test_compute_target_function():
    from yaml_bert.vocab import compute_target
    from yaml_bert.types import YamlNode, NodeType

    # Root key: unigram
    node = YamlNode("metadata", NodeType.KEY, depth=0, sibling_index=2, parent_path="")
    assert compute_target(node, "Deployment") == ("metadata", "simple")

    # Under metadata: bigram
    node = YamlNode("name", NodeType.KEY, depth=1, sibling_index=0, parent_path="metadata")
    assert compute_target(node, "Deployment") == ("metadata::name", "simple")

    # First level under spec: trigram
    node = YamlNode("replicas", NodeType.KEY, depth=1, sibling_index=0, parent_path="spec")
    assert compute_target(node, "Deployment") == ("Deployment::spec::replicas", "kind_specific")

    # Deeper under spec: bigram
    node = YamlNode("image", NodeType.LIST_KEY, depth=4, sibling_index=1, parent_path="spec.template.spec.containers.0")
    assert compute_target(node, "Deployment") == ("containers::image", "simple")

    # Under data (kind-specific root): trigram
    node = YamlNode("DB_HOST", NodeType.KEY, depth=1, sibling_index=0, parent_path="data")
    assert compute_target(node, "ConfigMap") == ("ConfigMap::data::DB_HOST", "kind_specific")

    # Under rules (kind-specific root): trigram
    node = YamlNode("apiGroups", NodeType.LIST_KEY, depth=1, sibling_index=0, parent_path="rules")
    assert compute_target(node, "ClusterRole") == ("ClusterRole::rules::apiGroups", "kind_specific")


def test_hybrid_vocab_saved_and_loaded(tmp_path):
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
""")
    annotator.annotate(nodes)
    vocab = VocabBuilder().build(nodes)

    path = str(tmp_path / "vocab.json")
    vocab.save(path)
    loaded = Vocabulary.load(path)

    assert loaded.simple_target_vocab == vocab.simple_target_vocab
    assert loaded.kind_target_vocab == vocab.kind_target_vocab
```

- [ ] **Step 2: Implement compute_target and hybrid vocabularies**

Add to `yaml_bert/vocab.py`:

```python
UNIVERSAL_ROOT_KEYS: set[str] = {"apiVersion", "kind", "metadata"}


def compute_target(node: YamlNode, kind: str) -> tuple[str, str]:
    """Compute the hybrid target for a node.

    Returns:
        (target_string, head_type) where head_type is "simple" or "kind_specific"
    """
    # Root keys: unigram
    if node.depth == 0:
        return node.token, "simple"

    parent_key: str = Vocabulary.extract_parent_key(node.parent_path)

    # First level under a kind-specific root key: trigram
    if node.depth == 1 and parent_key not in UNIVERSAL_ROOT_KEYS and parent_key != "":
        return f"{kind}::{parent_key}::{node.token}", "kind_specific"

    # Everything else: bigram
    return f"{parent_key}::{node.token}" if parent_key else node.token, "simple"
```

Update `Vocabulary.__init__` to accept `simple_target_vocab` and `kind_target_vocab`:

```python
def __init__(
    self,
    key_vocab, value_vocab, special_tokens,
    kind_vocab=None,
    simple_target_vocab=None,
    kind_target_vocab=None,
):
    # ... existing ...
    self.simple_target_vocab = simple_target_vocab or {}
    self.kind_target_vocab = kind_target_vocab or {}
```

Add encode/decode and size properties:

```python
def encode_simple_target(self, target: str) -> int:
    return self.simple_target_vocab.get(target, self.special_tokens["[UNK]"])

def encode_kind_target(self, target: str) -> int:
    return self.kind_target_vocab.get(target, self.special_tokens["[UNK]"])

@property
def simple_target_vocab_size(self) -> int:
    return len(self.simple_target_vocab) + len(self.special_tokens)

@property
def kind_target_vocab_size(self) -> int:
    return len(self.kind_target_vocab) + len(self.special_tokens)
```

Update `save` and `load` to include the new vocabs.

Update `VocabBuilder.build` to collect simple and kind targets:

```python
def build(self, nodes, min_freq=1):
    # ... existing key/value/kind collection ...

    # Collect hybrid targets
    simple_target_counts: dict[str, int] = {}
    kind_target_counts: dict[str, int] = {}
    current_kind: str = ""

    prev_was_kind_key: bool = False
    for node in nodes:
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            # Track kind for trigram targets
            if prev_was_kind_key:
                prev_was_kind_key = False
            if node.token == "kind" and node.depth == 0:
                prev_was_kind_key = True

            target, head_type = compute_target(node, current_kind)
            if head_type == "simple":
                simple_target_counts[target] = simple_target_counts.get(target, 0) + 1
            else:
                kind_target_counts[target] = kind_target_counts.get(target, 0) + 1
        elif node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE):
            if prev_was_kind_key:
                current_kind = node.token
                kind_set.add(node.token)
            prev_was_kind_key = False

    # Build target vocabs with offset
    simple_target_vocab = {
        t: i + offset
        for i, t in enumerate(sorted(t for t, c in simple_target_counts.items() if c >= min_freq))
    }
    kind_target_vocab = {
        t: i + offset
        for i, t in enumerate(sorted(t for t, c in kind_target_counts.items() if c >= min_freq))
    }

    return Vocabulary(key_vocab, value_vocab, special_tokens, kind_vocab,
                      simple_target_vocab, kind_target_vocab)
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_hybrid_vocab.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add yaml_bert/vocab.py tests/test_hybrid_vocab.py
git commit -m "feat: hybrid target vocabularies — bigram simple + trigram kind-specific"
```

---

### Task 2: Simplify embedding layer

Remove `kind_embedding` and `parent_key_embedding`. Keep 4 tables: token (key+value), depth, sibling, node_type.

**Files:**
- Modify: `yaml_bert/embedding.py`
- Modify: `tests/test_embedding.py`

- [ ] **Step 1: Create v4 embedding class**

Create `YamlBertEmbeddingV4` alongside the existing class (don't break v1-v3):

```python
class YamlBertEmbeddingV4(nn.Module):
    """v4 embedding: 4 tables (no kind_emb, no parent_key_emb)."""

    def __init__(
        self,
        config: YamlBertConfig,
        key_vocab_size: int,
        value_vocab_size: int,
    ) -> None:
        super().__init__()
        d: int = config.d_model
        self.key_embedding: nn.Embedding = nn.Embedding(key_vocab_size, d)
        self.value_embedding: nn.Embedding = nn.Embedding(value_vocab_size, d)
        self.depth_embedding: nn.Embedding = nn.Embedding(config.max_depth, d)
        self.sibling_embedding: nn.Embedding = nn.Embedding(config.max_sibling, d)
        self.node_type_embedding: nn.Embedding = nn.Embedding(4, d)
        self.layer_norm: nn.LayerNorm = nn.LayerNorm(d)

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
    ) -> torch.Tensor:
        is_key = (node_types == 0) | (node_types == 2)
        key_vocab_size = self.key_embedding.num_embeddings
        val_vocab_size = self.value_embedding.num_embeddings
        key_emb = self.key_embedding(token_ids.clamp(0, key_vocab_size - 1))
        val_emb = self.value_embedding(token_ids.clamp(0, val_vocab_size - 1))
        token_emb = torch.where(is_key.unsqueeze(-1), key_emb, val_emb)

        tree_pos = (
            self.depth_embedding(depths)
            + self.sibling_embedding(sibling_indices)
            + self.node_type_embedding(node_types)
        )

        return self.layer_norm(token_emb + tree_pos)
```

- [ ] **Step 2: Add tests, commit**

```bash
git add yaml_bert/embedding.py tests/test_embedding.py
git commit -m "feat: YamlBertEmbeddingV4 — 4 tables, no kind/parent_key embedding"
```

---

### Task 3: v4 model with two prediction heads

**Files:**
- Modify: `yaml_bert/model.py`
- Create: `tests/test_model_v4.py`

- [ ] **Step 1: Create YamlBertModelV4**

```python
class YamlBertModelV4(nn.Module):
    def __init__(
        self,
        config: YamlBertConfig,
        embedding: YamlBertEmbeddingV4,
        simple_vocab_size: int,
        kind_vocab_size: int,
    ) -> None:
        super().__init__()
        self.embedding = embedding

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model, nhead=config.num_heads,
            dim_feedforward=config.d_ff, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.simple_head = nn.Linear(config.d_model, simple_vocab_size)
        self.kind_head = nn.Linear(config.d_model, kind_vocab_size)

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=-100)

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embedding(token_ids, node_types, depths, sibling_indices)
        x = self.encoder(x, src_key_padding_mask=padding_mask)
        simple_logits = self.simple_head(x)
        kind_logits = self.kind_head(x)
        return simple_logits, kind_logits

    def compute_loss(
        self,
        simple_logits: torch.Tensor,
        simple_labels: torch.Tensor,
        kind_logits: torch.Tensor,
        kind_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        simple_loss = self.loss_fn(
            simple_logits.view(-1, simple_logits.size(-1)),
            simple_labels.view(-1),
        )
        kind_loss = self.loss_fn(
            kind_logits.view(-1, kind_logits.size(-1)),
            kind_labels.view(-1),
        )
        total = simple_loss + kind_loss
        return total, {"simple": simple_loss.item(), "kind": kind_loss.item()}
```

- [ ] **Step 2: Write tests**

File: `tests/test_model_v4.py`

Test output shapes, loss computation, two-head return values.

- [ ] **Step 3: Commit**

```bash
git add yaml_bert/model.py tests/test_model_v4.py
git commit -m "feat: YamlBertModelV4 with simple + kind-specific prediction heads"
```

---

### Task 4: Update dataset for hybrid labels

**Files:**
- Modify: `yaml_bert/dataset.py`
- Modify: `tests/test_dataset.py`

- [ ] **Step 1: Update __getitem__ to produce simple_labels and kind_labels**

For each masked KEY node:
1. Call `compute_target(node, kind)` to get the target and head type
2. If head_type == "simple": set `simple_labels[i]` to the encoded simple target, `kind_labels[i]` = -100
3. If head_type == "kind_specific": set `kind_labels[i]` to the encoded kind target, `simple_labels[i]` = -100

The return dict changes from:
```python
{"token_ids", "node_types", "depths", "sibling_indices", "parent_key_ids", "labels", "kind_ids"}
```

To (for v4):
```python
{"token_ids", "node_types", "depths", "sibling_indices", "simple_labels", "kind_labels"}
```

Note: no `parent_key_ids` and no `kind_ids` — those are removed from v4 input.

- [ ] **Step 2: Add a `from_cached_docs_v4` classmethod** that builds the v4 dataset

- [ ] **Step 3: Tests and commit**

```bash
git add yaml_bert/dataset.py tests/test_dataset.py
git commit -m "feat: v4 dataset with hybrid simple/kind labels"
```

---

### Task 5: Update trainer for v4

**Files:**
- Modify: `yaml_bert/trainer.py`

- [ ] **Step 1: Create YamlBertTrainerV4** or update existing trainer to handle v4 model

The training loop:
```python
simple_logits, kind_logits = self.model(
    token_ids=batch["token_ids"],
    node_types=batch["node_types"],
    depths=batch["depths"],
    sibling_indices=batch["sibling_indices"],
    padding_mask=batch["padding_mask"],
)

loss, breakdown = self.model.compute_loss(
    simple_logits, batch["simple_labels"],
    kind_logits, batch["kind_labels"],
)
```

tqdm shows: `loss=X.XX, simple=X.XX, kind=X.XX`

- [ ] **Step 2: Commit**

```bash
git add yaml_bert/trainer.py
git commit -m "feat: v4 trainer with hybrid loss"
```

---

### Task 6: v4 training script

**Files:**
- Create: `scripts/train_v4.py`

- [ ] **Step 1: Create training script**

Similar to `train_hf.py` but uses:
- `YamlBertEmbeddingV4` (4 tables)
- `YamlBertModelV4` (two heads)
- `from_cached_docs_v4` dataset
- Hybrid loss

Uses the doc_cache for fast loading. Builds hybrid target vocab from cached docs.

CLI:
```bash
python scripts/train_v4.py --max-docs 0 --epochs 15 --vocab-min-freq 100 --output-dir output_v4
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_v4.py
git commit -m "feat: v4 training script with hybrid targets"
```

---

### Task 7: Quick validation on 5K docs

- [ ] **Step 1: Train on 5K docs**

```bash
PYTHONPATH=. python scripts/train_v4.py --max-docs 5000 --epochs 10 --vocab-min-freq 10 --output-dir output_v4_quick
```

- [ ] **Step 2: Check loss breakdown**

Both simple and kind losses should be meaningful (not near-zero like v3's auxiliary losses).

- [ ] **Step 3: Test CRD generalization**

Feed the model a made-up kind and check:
- Does it predict metadata fields correctly? (universal bigrams)
- Does it predict spec fields with low confidence? (unseen trigrams)

- [ ] **Step 4: Compare document embedding similarity**

Run the same 4-document cosine similarity test (Deployment vs Service vs Pod vs ConfigMap). Target: cosine similarity < 0.7 (vs current 0.89).

- [ ] **Step 5: Commit results**

```bash
git add -A
git commit -m "feat: v4 validation results on 5K docs"
```
