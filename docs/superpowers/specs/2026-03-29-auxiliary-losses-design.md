# YAML-BERT v3: Auxiliary Losses for Structural Preservation

## Problem

Probing analysis of v1/v2 revealed that tree positional information degrades through transformer layers:

| Property | Embedding Input | After Layer 5 | Loss |
|----------|:-:|:-:|:-:|
| depth | 87% | 67% | -20% |
| parent_key | 79% | 63% | -16% |

The model achieves 95% key prediction accuracy but discards structural information it doesn't need for key prediction. This causes failures on kind-specific anomaly detection — the model can't tell that `replicas` is valid under `spec` in a Deployment but not in a Pod.

**Root cause:** The model only optimizes key prediction loss. It learns what the loss rewards and discards everything else.

## Solution

Add auxiliary classification losses at the final transformer layer that force the model to preserve kind and parent_key information.

```
total_loss = key_prediction_loss + α * kind_classification_loss + β * parent_key_classification_loss
```

Two small linear classifiers on the final layer output:
- Kind classifier: `Linear(d_model, kind_vocab_size)` — "what resource kind is this document?"
- Parent key classifier: `Linear(d_model, key_vocab_size)` — "who is this node's parent?"

## Why These Two Properties

**Kind** — the specific identified gap. All 4 remaining anomaly detection failures involve kind-specific constraints where parent_key is identical (`spec`). Only kind differentiates valid from invalid structure.

**Parent key** — degrades 16% through layers. Preserving parent awareness strengthens the model's understanding of tree structure generally. While not needed for the specific 4 test failures, it improves structural awareness for future test cases and use cases.

**Not depth** — parent_key implicitly encodes depth. A node with `parent_key=containers` is necessarily deeper than one with `parent_key=spec`. Adding depth auxiliary loss would be redundant.

## Architecture

### New Components in YamlBertModel

```python
self.key_prediction_head: nn.Linear      # d_model → key_vocab_size (existing)
self.kind_classifier: nn.Linear          # d_model → kind_vocab_size (NEW)
self.parent_key_classifier: nn.Linear    # d_model → key_vocab_size (NEW)
```

### Forward Pass

```python
def forward(self, ...):
    x = self.embedding(...)
    x = self.encoder(x, ...)
    key_logits = self.key_prediction_head(x)
    kind_logits = self.kind_classifier(x)
    parent_logits = self.parent_key_classifier(x)
    return key_logits, kind_logits, parent_logits
```

### Loss Computation

```python
def compute_loss(self, key_logits, labels, kind_logits, kind_labels,
                 parent_logits, parent_labels, alpha, beta):
    key_loss = CrossEntropyLoss(ignore_index=-100)(
        key_logits.view(-1, key_logits.size(-1)), labels.view(-1)
    )
    kind_loss = CrossEntropyLoss()(
        kind_logits.view(-1, kind_logits.size(-1)), kind_labels.view(-1)
    )
    parent_loss = CrossEntropyLoss()(
        parent_logits.view(-1, parent_logits.size(-1)), parent_labels.view(-1)
    )
    return key_loss + alpha * kind_loss + beta * parent_loss
```

Kind and parent losses are computed on **all nodes**, not just masked positions. Every node must preserve kind and parent_key info.

## Config Changes

```python
@dataclass
class YamlBertConfig:
    # ... existing fields ...
    aux_kind_weight: float = 0.1      # α
    aux_parent_weight: float = 0.1    # β
```

Exposed as `--alpha` and `--beta` CLI flags in `train_hf.py`.

### Calibration

Start with α=0.1, β=0.1. After epoch 1, check loss magnitudes:
- Auxiliary losses should contribute ~10-20% of total loss
- If too small (model ignores them), increase weights
- If too large (key prediction suffers), decrease weights

## Data Changes

**None.** The dataset already provides `kind_ids` and `parent_key_ids` in every batch. These become the labels for the auxiliary classifiers.

## Files Changed

| File | Change |
|------|--------|
| `yaml_bert/config.py` | Add `aux_kind_weight`, `aux_parent_weight` |
| `yaml_bert/model.py` | Add two classifier heads, update `forward` and `compute_loss` |
| `yaml_bert/trainer.py` | Pass auxiliary labels and weights to `compute_loss` |
| `scripts/train_hf.py` | Add `--alpha` and `--beta` CLI flags |
| `tests/test_model.py` | Test new forward return values and loss |

## Success Criteria

After training v3:

1. **Probing**: parent_key accuracy at final layer significantly above 63%
2. **Probing**: kind accuracy at final layer near 100%
3. **Key prediction**: accuracy remains ≥95%
4. **Anomaly detection**: improvement on the 4 remaining kind-specific failures
5. **Capability tests**: ≥71/77 (v2 baseline), ideally >74/77

## What This Does NOT Change

- Embedding layer unchanged (kind_embedding from v2 stays)
- Dataset unchanged (already provides kind_ids and parent_key_ids)
- Attention mechanism unchanged
- Masking strategy unchanged (key-only)
