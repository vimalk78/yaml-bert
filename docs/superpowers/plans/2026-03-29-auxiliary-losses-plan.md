# Auxiliary Losses Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add kind and parent_key auxiliary classification losses to force the model to preserve tree structural information through all transformer layers.

**Architecture:** Two new linear classifiers on the final layer output, with configurable loss weights α (kind) and β (parent_key). Total loss = key_prediction + α * kind_classification + β * parent_key_classification. No changes to embedding, dataset, or attention.

**Tech Stack:** Python 3.10+, PyTorch >= 2.0

**Spec:** `docs/superpowers/specs/2026-03-29-auxiliary-losses-design.md`

---

### Task 1: Add auxiliary loss weights to config

**Files:**
- Modify: `yaml_bert/config.py`
- Modify: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_config.py`:

```python
def test_auxiliary_loss_weights():
    config = YamlBertConfig()
    assert config.aux_kind_weight == 0.1
    assert config.aux_parent_weight == 0.1

    config2 = YamlBertConfig(aux_kind_weight=0.5, aux_parent_weight=0.0)
    assert config2.aux_kind_weight == 0.5
    assert config2.aux_parent_weight == 0.0
```

- [ ] **Step 2: Add fields to YamlBertConfig**

Add after `max_seq_len` in `yaml_bert/config.py`:

```python
    # Auxiliary loss weights
    aux_kind_weight: float = 0.1     # α: kind classification loss weight
    aux_parent_weight: float = 0.1   # β: parent_key classification loss weight
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_config.py -v`
Expected: ALL PASS

- [ ] **Step 4: Commit**

```bash
git add yaml_bert/config.py tests/test_config.py
git commit -m "feat: add aux_kind_weight and aux_parent_weight to config"
```

---

### Task 2: Add auxiliary classifiers to model

**Files:**
- Modify: `yaml_bert/model.py`
- Modify: `tests/test_model.py`

- [ ] **Step 1: Write failing test**

Append to `tests/test_model.py`:

```python
def test_model_returns_auxiliary_logits():
    config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)
    emb = YamlBertEmbedding(
        config=config, key_vocab_size=100, value_vocab_size=200, kind_vocab_size=10,
    )
    model = YamlBertModel(
        config=config, embedding=emb, key_vocab_size=100, kind_vocab_size=10,
    )

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))
    kind_ids = torch.ones(batch_size, seq_len, dtype=torch.long)

    key_logits, kind_logits, parent_logits = model(
        token_ids, node_types, depths, siblings, parent_keys, kind_ids=kind_ids,
    )

    assert key_logits.shape == (batch_size, seq_len, 100)
    assert kind_logits.shape == (batch_size, seq_len, 10)
    assert parent_logits.shape == (batch_size, seq_len, 100)


def test_model_auxiliary_loss():
    config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)
    emb = YamlBertEmbedding(
        config=config, key_vocab_size=100, value_vocab_size=200, kind_vocab_size=10,
    )
    model = YamlBertModel(
        config=config, embedding=emb, key_vocab_size=100, kind_vocab_size=10,
    )

    batch_size = 2
    seq_len = 5
    token_ids = torch.randint(0, 50, (batch_size, seq_len))
    node_types = torch.zeros(batch_size, seq_len, dtype=torch.long)
    depths = torch.randint(0, 5, (batch_size, seq_len))
    siblings = torch.randint(0, 3, (batch_size, seq_len))
    parent_keys = torch.randint(0, 50, (batch_size, seq_len))
    kind_ids = torch.ones(batch_size, seq_len, dtype=torch.long)

    labels = torch.full((batch_size, seq_len), -100, dtype=torch.long)
    labels[0, 1] = 10

    key_logits, kind_logits, parent_logits = model(
        token_ids, node_types, depths, siblings, parent_keys, kind_ids=kind_ids,
    )

    loss = model.compute_loss(
        key_logits=key_logits,
        labels=labels,
        kind_logits=kind_logits,
        kind_labels=kind_ids,
        parent_logits=parent_logits,
        parent_labels=parent_keys,
        alpha=0.1,
        beta=0.1,
    )

    assert loss.dim() == 0
    assert loss.item() > 0
    assert loss.requires_grad
```

- [ ] **Step 2: Update YamlBertModel**

Modify `yaml_bert/model.py`:

```python
class YamlBertModel(nn.Module):
    def __init__(
        self,
        config: YamlBertConfig,
        embedding: YamlBertEmbedding,
        key_vocab_size: int,
        kind_vocab_size: int | None = None,
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
        self.kind_classifier: nn.Linear = nn.Linear(config.d_model, kind_vocab_size or 1)
        self.parent_key_classifier: nn.Linear = nn.Linear(config.d_model, key_vocab_size)

        self.key_loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=-100)
        self.aux_loss_fn: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    def forward(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        parent_key_ids: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
        kind_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x: torch.Tensor = self.embedding(
            token_ids, node_types, depths, sibling_indices, parent_key_ids,
            kind_ids=kind_ids,
        )
        x = self.encoder(x, src_key_padding_mask=padding_mask)

        key_logits: torch.Tensor = self.key_prediction_head(x)
        kind_logits: torch.Tensor = self.kind_classifier(x)
        parent_logits: torch.Tensor = self.parent_key_classifier(x)

        return key_logits, kind_logits, parent_logits

    def compute_loss(
        self,
        key_logits: torch.Tensor,
        labels: torch.Tensor,
        kind_logits: torch.Tensor | None = None,
        kind_labels: torch.Tensor | None = None,
        parent_logits: torch.Tensor | None = None,
        parent_labels: torch.Tensor | None = None,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> torch.Tensor:
        key_loss: torch.Tensor = self.key_loss_fn(
            key_logits.view(-1, key_logits.size(-1)),
            labels.view(-1),
        )

        total_loss: torch.Tensor = key_loss

        if alpha > 0 and kind_logits is not None and kind_labels is not None:
            kind_loss: torch.Tensor = self.aux_loss_fn(
                kind_logits.view(-1, kind_logits.size(-1)),
                kind_labels.view(-1),
            )
            total_loss = total_loss + alpha * kind_loss

        if beta > 0 and parent_logits is not None and parent_labels is not None:
            parent_loss: torch.Tensor = self.aux_loss_fn(
                parent_logits.view(-1, parent_logits.size(-1)),
                parent_labels.view(-1),
            )
            total_loss = total_loss + beta * parent_loss

        return total_loss

    @torch.no_grad()
    def get_attention_weights(
        self,
        token_ids: torch.Tensor,
        node_types: torch.Tensor,
        depths: torch.Tensor,
        sibling_indices: torch.Tensor,
        parent_key_ids: torch.Tensor,
        kind_ids: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """Extract attention weights from all layers."""
        self.eval()
        x: torch.Tensor = self.embedding(
            token_ids, node_types, depths, sibling_indices, parent_key_ids,
            kind_ids=kind_ids,
        )

        attention_weights: list[torch.Tensor] = []
        for layer in self.encoder.layers:
            attn_output, attn_weight = layer.self_attn(
                x, x, x, need_weights=True, average_attn_weights=False
            )
            attention_weights.append(attn_weight)
            x = layer(x)

        return attention_weights
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/test_model.py -v`
Expected: ALL PASS (old tests need updating — `forward` now returns 3 tensors)

NOTE: Old tests that do `key_logits = model(...)` need to change to `key_logits, _, _ = model(...)`. Update `test_model_output_shape`, `test_model_with_padding_mask`, `test_model_loss_computation`, `test_model_with_kind_ids`, and `test_model_without_kind_ids_backward_compatible`.

- [ ] **Step 4: Run all tests**

Run: `pytest tests/ -q --ignore=tests/test_trainer.py --ignore=tests/test_e2e.py --ignore=tests/test_evaluate.py`
Expected: ALL PASS

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/model.py tests/test_model.py
git commit -m "feat: add kind and parent_key auxiliary classifiers to YamlBertModel"
```

---

### Task 3: Update trainer to use auxiliary losses

**Files:**
- Modify: `yaml_bert/trainer.py`

- [ ] **Step 1: Update training loop**

Change the model call and loss computation in `yaml_bert/trainer.py`:

```python
                key_logits, kind_logits, parent_logits = self.model(
                    token_ids=batch["token_ids"],
                    node_types=batch["node_types"],
                    depths=batch["depths"],
                    sibling_indices=batch["sibling_indices"],
                    parent_key_ids=batch["parent_key_ids"],
                    padding_mask=batch["padding_mask"],
                    kind_ids=batch.get("kind_ids"),
                )

                loss: torch.Tensor = self.model.compute_loss(
                    key_logits=key_logits,
                    labels=batch["labels"],
                    kind_logits=kind_logits,
                    kind_labels=batch.get("kind_ids"),
                    parent_logits=parent_logits,
                    parent_labels=batch.get("parent_key_ids"),
                    alpha=self.config.aux_kind_weight,
                    beta=self.config.aux_parent_weight,
                )
```

- [ ] **Step 2: Commit**

```bash
git add yaml_bert/trainer.py
git commit -m "feat: pass auxiliary labels and weights in trainer"
```

---

### Task 4: Update evaluator and scripts

All scripts that call `model(...)` and expect a single tensor need to unpack 3 tensors. Also update `model.compute_loss` calls.

**Files:**
- Modify: `yaml_bert/evaluate.py`
- Modify: `scripts/anomaly_score.py`
- Modify: `scripts/evaluate_checkpoint.py`
- Modify: `scripts/evaluate_all.py`
- Modify: `model_tests/test_structural.py`
- Modify: `model_tests/test_capabilities.py`

- [ ] **Step 1: Update evaluate.py**

In `evaluate_prediction_accuracy` and `top_k_predictions`, change:
```python
key_logits = self.model(...)
```
to:
```python
key_logits, _, _ = self.model(...)
```

- [ ] **Step 2: Update anomaly_score.py**

In `score_yaml`, change:
```python
logits = model(...)
```
to:
```python
logits, _, _ = model(...)
```

- [ ] **Step 3: Update all other scripts**

Same pattern for `evaluate_checkpoint.py`, `evaluate_all.py`, `test_structural.py`, `test_capabilities.py` — anywhere `model(...)` is called, unpack 3 return values, keep only `key_logits`.

- [ ] **Step 4: Run capability tests on v1 checkpoint to verify backward compat**

Run: `CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python model_tests/test_capabilities.py output_v1/yaml_bert_v1_final.pt --vocab output_v1/vocab.json`
Expected: 68/77 (same as before)

- [ ] **Step 5: Commit**

```bash
git add yaml_bert/evaluate.py scripts/ model_tests/
git commit -m "feat: update all scripts for 3-tuple model output"
```

---

### Task 5: Add --alpha and --beta CLI flags to training script

**Files:**
- Modify: `scripts/train_hf.py`

- [ ] **Step 1: Add CLI arguments**

In `parse_args()`, add:

```python
    parser.add_argument("--alpha", type=float, default=None,
                        help="Kind auxiliary loss weight (default: from config)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Parent key auxiliary loss weight (default: from config)")
```

After config creation, apply overrides:

```python
    if args.alpha is not None:
        config.aux_kind_weight = args.alpha
    if args.beta is not None:
        config.aux_parent_weight = args.beta
```

Also add `kind_vocab_size` to the `YamlBertModel` constructor:

```python
    model: YamlBertModel = YamlBertModel(
        config=config,
        embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
```

Also update `scripts/train.py` with the same changes.

- [ ] **Step 2: Commit**

```bash
git add scripts/train_hf.py scripts/train.py
git commit -m "feat: add --alpha and --beta CLI flags for auxiliary loss weights"
```

---

### Task 6: Train v3 and evaluate

- [ ] **Step 1: Train v3**

```bash
python scripts/train_hf.py --max-docs 0 --full --epochs 15 --vocab-min-freq 100 --alpha 0.1 --beta 0.1 --output-dir output_v3
```

- [ ] **Step 2: After epoch 1, check loss magnitudes**

Look at the printed loss. If auxiliary losses dominate, adjust alpha/beta and restart.

- [ ] **Step 3: Run probing on v3 checkpoint**

```bash
CUDA_VISIBLE_DEVICES="" PYTHONPATH=. python scripts/probe_heads.py output_v3/checkpoints/yaml_bert_epoch_5.pt --vocab output_v3/vocab.json --output-dir output_v3/probes
```

Compare with v1/v2: parent_key at final layer should be above 63%.

- [ ] **Step 4: Run capability tests**

```bash
PYTHONPATH=. python model_tests/test_capabilities.py output_v3/checkpoints/yaml_bert_epoch_5.pt --vocab output_v3/vocab.json
```

Target: >71/77 (v2 baseline).

- [ ] **Step 5: Run anomaly detection**

```bash
PYTHONPATH=. python scripts/anomaly_score.py output_v3/checkpoints/yaml_bert_epoch_5.pt --vocab output_v3/vocab.json --run-examples
```

Target: >2/10 (v1 baseline) detected.

- [ ] **Step 6: Export final model and tag**

```bash
python scripts/export_model.py output_v3/checkpoints/yaml_bert_epoch_X.pt --output output_v3/yaml_bert_v3_final.pt
git tag -a v3.0 -m "v3: Auxiliary losses for structural preservation"
```
