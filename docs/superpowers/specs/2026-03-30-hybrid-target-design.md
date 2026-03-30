# YAML-BERT v4: Hybrid Prediction Targets

## Problem

The model produces similar hidden states across different resource types (0.84-0.92 cosine similarity) because the prediction target is just the key name. `metadata.name` predicts "name" regardless of kind — the model has no incentive to produce kind-specific representations.

Compound targets ("Deployment.metadata.name") fix this but over-correct: they force kind-specific representations for universal structure (TypeMeta, ObjectMeta) that should be kind-agnostic.

## Insight from K8s Object Model

Every K8s resource has the same Go struct pattern:

```go
type PowerMonitor struct {
    metav1.TypeMeta   `json:",inline"`      // kind, apiVersion — UNIVERSAL
    metav1.ObjectMeta `json:"metadata"`     // name, namespace, labels — UNIVERSAL
    Spec   PowerMonitorSpec   `json:"spec"` // KIND-SPECIFIC
    Status PowerMonitorStatus `json:"status"` // KIND-SPECIFIC
}
```

- `TypeMeta` and `ObjectMeta` are embedded — identical across ALL resource types
- `Spec` and `Status` are type-specific — different for every kind

The prediction target should follow this structure.

## Solution: Hybrid Prediction Targets

Use simple targets for universal structure, compound targets for kind-specific structure:

| Tree region | Example nodes | Target format | Example target |
|---|---|---|---|
| Root keys | apiVersion, kind, metadata, spec | Simple key | `"metadata"` |
| metadata.* | name, namespace, labels, annotations | Simple key | `"name"` |
| spec.* | replicas, selector, containers, template | Compound (kind.key) | `"Deployment.replicas"` |
| status.* | availableReplicas, conditions | Compound (kind.key) | `"Deployment.availableReplicas"` |
| Nested under spec | matchLabels, containerPort, image | Compound (kind.parent.key) | `"Deployment.containers.image"` |

### Why This Works

- **Universal keys** (`metadata.name`, `metadata.labels`): same target across all kinds → model learns a single "name" representation that generalizes to any CRD
- **Kind-specific keys** (`spec.replicas`, `spec.containers`): different target per kind → model MUST produce different hidden states for `replicas` in Deployment vs Pod
- **Follows K8s architecture**: not a heuristic — mirrors the actual TypeMeta/ObjectMeta/Spec/Status separation

### The Math

For `metadata.name`:
- Target = "name" (same in Deployment, Service, Pod)
- Optimal hidden state: same across kinds ✓
- Generalizes to new CRDs ✓

For `spec.replicas`:
- Target = "Deployment.replicas" vs "StatefulSet.replicas" vs absent in Pod
- Optimal hidden states MUST differ because targets differ ✓
- Kind-specific discrimination ✓

## Input Embeddings

Remove redundant components:

| Embedding | Keep? | Reason |
|---|---|---|
| token_emb | Yes | Base token identity |
| depth_emb | Yes | Not in any target |
| sibling_emb | Yes | Not in any target |
| node_type_emb | Yes | Not in any target |
| parent_key_emb | Remove | In compound target for spec/status nodes |
| kind_emb | Remove | In compound target for spec/status nodes |

```
input = token_emb + depth_emb + sibling_emb + type_emb
```

The model learns parent and kind awareness from the hybrid target gradient — no shortcut through input embedding. For metadata nodes (simple target), the model doesn't need kind info. For spec nodes (compound target), the model must learn kind from attention to the kind value node.

## Vocabulary

Two prediction heads:

**Simple head**: predicts from the original key vocabulary (~1664 classes). Used for root keys and metadata nodes.

**Compound head**: predicts from a (kind.key) or (kind.parent.key) vocabulary. Size = number of unique compound tokens in training data. Estimated ~5000-10000 classes.

### How to determine which head to use

During training and inference, the node's `parent_path` determines the head:
- If `parent_path` is empty or starts with `metadata`: use simple head
- Otherwise (under `spec`, `status`, or any other kind-specific subtree): use compound head

This is a static rule based on tree position, not a learned decision.

## Architecture Changes

```python
class YamlBertModel(nn.Module):
    def __init__(self, ...):
        self.embedding  # 4 tables: token, depth, sibling, node_type
        self.encoder    # same TransformerEncoder
        self.simple_head: nn.Linear   # d_model → simple_vocab_size (~1664)
        self.compound_head: nn.Linear  # d_model → compound_vocab_size (~8000)

    def forward(self, ...):
        x = self.embedding(...)
        x = self.encoder(x)
        simple_logits = self.simple_head(x)
        compound_logits = self.compound_head(x)
        return simple_logits, compound_logits
```

Loss computation:
```python
# For each masked position, use the appropriate head based on tree position
simple_mask = is_metadata_or_root(parent_paths)
compound_mask = ~simple_mask

simple_loss = CrossEntropyLoss(simple_logits[simple_mask], simple_labels[simple_mask])
compound_loss = CrossEntropyLoss(compound_logits[compound_mask], compound_labels[compound_mask])

total_loss = simple_loss + compound_loss
```

## Training

Same as before: mask 15% of KEY nodes, predict the target. The only change is the target format and which head produces the prediction.

The dataset needs to generate two label tensors:
- `simple_labels`: original key token IDs (for metadata/root nodes)
- `compound_labels`: compound token IDs (for spec/status nodes)
- Non-masked positions and positions using the other head get -100 (ignore)

## Expected Outcomes

1. **Key prediction accuracy**: should remain ~95% for simple targets, may be lower for compound targets initially (larger vocabulary)
2. **Document embeddings**: significantly more discriminative because spec/status hidden states are kind-specific
3. **Kind-specific rejection**: `replicas` in Pod spec should now be low confidence because the compound target "Pod.replicas" is rare/absent in training
4. **Generalization**: universal structure (metadata) generalizes to unseen CRDs. Kind-specific structure doesn't — but that's correct

## Files Changed

| File | Change |
|------|--------|
| `yaml_bert/embedding.py` | Remove kind_emb and parent_key_emb |
| `yaml_bert/model.py` | Two prediction heads (simple + compound) |
| `yaml_bert/vocab.py` | Build compound vocabulary |
| `yaml_bert/dataset.py` | Generate simple_labels and compound_labels |
| `yaml_bert/trainer.py` | Compute hybrid loss |
| `yaml_bert/config.py` | No new hyperparameters needed |

## What This Does NOT Change

- Transformer encoder architecture (same layers, heads, d_model)
- Masking strategy (15% key-only)
- Training data (same 276K corpus)
- Evaluation framework (capability tests, probing, suggest tool)
