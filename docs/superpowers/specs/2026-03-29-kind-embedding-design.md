# Kind Embedding: Document-Level Positional Component

## Problem

The model learned local tree structure (parent-child relationships) but fails to enforce kind-specific constraints. When predicting keys under `spec`, it predicts `replicas` with 99% confidence regardless of whether the document is a Deployment (valid) or a Pod (invalid).

The root cause: every node's positional encoding is local — depth, sibling, node_type, parent_key. No component tells the node "you're inside a Deployment" vs "you're inside a Service."

The `kind` value IS in the sequence as an unmasked value token, but attention patterns are mostly local. Deep nodes don't attend to `kind` at the top of the document.

## Solution

Add `kind_embedding` as a document-level positional component. Every node in the document gets the same kind embedding added to its input vector.

```
tree_pos = depth_emb + sibling_emb + type_emb + parent_key_emb + kind_emb
                                                                   ↑ NEW
```

Analogous to BERT's segment embedding — a document-level property baked into every token.

## What Changes

### Embedding Layer

New embedding table:

| Table | Size | Index | Role |
|-------|------|-------|------|
| `kind_embedding` | kind_vocab_size x d_model | kind token ID | "I am inside this kind of resource" — Deployment, Service, Pod, etc. |

`kind_vocab_size` uses a dedicated vocabulary built from the `kind` values seen in the training corpus. Not the key or value vocab — a separate small vocab of resource types.

### Data Flow

During linearization/dataset creation:
1. Extract the `kind` value from each YAML document (always at root level)
2. Encode it as an integer ID via the kind vocabulary
3. Every node in that document gets the same `kind_id`
4. The `kind_id` is passed as an additional tensor to the model

### Formula

Before:
```
input(node) = LayerNorm(token_emb + depth_emb + sibling_emb + type_emb + parent_key_emb)
```

After:
```
input(node) = LayerNorm(token_emb + depth_emb + sibling_emb + type_emb + parent_key_emb + kind_emb)
```

### What This Fixes

With kind embedding, the anomaly scorer would see:

| Example | Before (no kind_emb) | After (with kind_emb) |
|---------|---------------------|----------------------|
| `replicas` under spec in Deployment | 99% confident | 99% confident (correct) |
| `replicas` under spec in Pod | 99% confident (WRONG) | Low confidence (correct) |
| `containers` directly under spec in Deployment | 99% confident (WRONG) | Low confidence (correct) |
| `containers` under spec in Pod | 99% confident | 99% confident (correct) |

The same key at the same depth under the same parent gets different predictions based on the document's kind.

## Files Changed

| File | Change |
|------|--------|
| `yaml_bert/config.py` | No change needed |
| `yaml_bert/vocab.py` | Add kind vocabulary building (small — ~50 unique kinds) |
| `yaml_bert/embedding.py` | Add `kind_embedding` table, accept `kind_ids` in forward |
| `yaml_bert/model.py` | Pass `kind_ids` through forward |
| `yaml_bert/dataset.py` | Extract `kind` from each document, include `kind_ids` in tensors |
| `yaml_bert/dataset.py` | Update `collate_fn` to pad `kind_ids` |
| `yaml_bert/trainer.py` | Pass `kind_ids` in batch |
| `yaml_bert/evaluate.py` | Pass `kind_ids` in evaluation |

## Kind Vocabulary

Built from training corpus. Expected to be small (~30-50 entries):

```
Deployment, Service, Pod, ConfigMap, Secret, Namespace,
StatefulSet, DaemonSet, Job, CronJob, Ingress, NetworkPolicy,
ClusterRole, ClusterRoleBinding, Role, RoleBinding,
ServiceAccount, PersistentVolumeClaim, PersistentVolume,
HorizontalPodAutoscaler, PodDisruptionBudget, StorageClass,
CustomResourceDefinition, ReplicaSet, ResourceQuota, LimitRange,
ValidatingWebhookConfiguration, Endpoints, ...
```

Documents with unknown or missing `kind` get `[UNK]` — the model treats them as generic.

## Checkpoint Compatibility

The `kind_embedding` is **optional**. Implementation uses `strict=False` when loading state dicts and defaults to `None` if not present:

- V1 checkpoints load into new code — `kind_embedding` weights are randomly initialized but unused (no `kind_ids` passed)
- V2 checkpoints load fully — `kind_embedding` weights are loaded
- Old calling code that doesn't pass `kind_ids` still works — kind embedding is skipped
- Existing unit tests pass without changes
- Model tests (`test_capabilities.py`, `test_structural.py`) work with both v1 and v2 checkpoints

No breaking changes.

## Design Decisions

1. **Kind extracted during dataset creation** — keeps `YamlNode` unchanged. Kind is a document-level property, not a per-node property. The dataset extracts it from the raw node list (find the node where `token="kind"` and take the next VALUE node's token).

2. **Documents without `kind` use `[UNK]`** — no new special token. Adding a new special token like `[NO_KIND]` would shift all vocab IDs and break v1 compatibility. `[UNK]` (ID=1) already exists and serves the purpose.

## New Visualizations

### Kind embedding table visualization
- Plot the kind embedding vectors for all ~50 kinds using t-SNE/PCA to 2D
- Similar kinds should cluster (Deployment/StatefulSet/DaemonSet are all workloads, Service/Ingress are networking, Role/ClusterRole are RBAC)
- Compare with: are the clusters semantically meaningful?

### Tree visualization with kind
- Extend `visualize_tree.py` to show a `_kind.png` component view
- All nodes same color (since kind is document-level, every node gets the same kind_emb)
- More useful: show TWO trees side by side — same YAML structure but different `kind` values — showing how the full embedding changes

### Kind effect heatmap
- For each (kind, parent_key) combination, show the model's top-3 predicted keys
- Visualize as a matrix: kinds on rows, parent keys on columns, predicted keys in cells
- Shows how kind conditions structure predictions

### Before/after comparison
- Run the same anomaly examples on v1 (no kind_emb) and v2 (with kind_emb)
- Side-by-side comparison of confidence scores
- Proves that kind embedding fixes the detection failures

## New Capability Tests

### Capability: Kind-specific spec children (expand existing)

Additional test cases beyond what's in the current suite:

```
- Pod spec has containers (not template)
- Deployment spec has template (not containers directly)
- StatefulSet spec has volumeClaimTemplates
- DaemonSet spec has updateStrategy (not replicas)
- Job spec has backoffLimit and template
- CronJob spec has schedule and jobTemplate
- Service spec has ports, selector, type (not containers)
- ConfigMap has data (not spec)
- Secret has data and type (not spec)
- PVC spec has accessModes, resources
- Namespace has no spec
```

### Capability: Kind-specific invalid structure rejection

These are the anomaly cases that v1 fails. v2 should catch them:

```
- replicas in Pod spec → should flag (Pods don't have replicas)
- replicas in Service spec → should flag
- replicas in ConfigMap → should flag
- containers directly under Deployment spec → should flag (needs template)
- template in Pod spec → should flag (Pods have containers directly)
- ports in Deployment spec → should flag (ports go in containers or Service)
- selector in ConfigMap → should flag
- data in Deployment → should flag
- spec in ConfigMap → should flag (uses data, not spec)
- spec in Secret → should flag (uses data, not spec)
- schedule in Deployment → should flag (CronJob field)
- serviceName in Deployment → should flag (StatefulSet field)
```

### Capability: Same structure, different kind

Test that the model gives different predictions for identical tree positions when the `kind` differs:

```
- Mask first key under spec in Deployment → expect replicas/selector/template
- Mask first key under spec in Service → expect type/ports/selector
- Mask first key under spec in Pod → expect containers/volumes/nodeSelector
- Mask first key under spec in Job → expect template/backoffLimit/completions
```

Same depth, same parent_key (spec), same sibling_index — only kind differs.

### Capability: Kind embedding does not harm valid structures

Verify that adding kind embedding doesn't reduce accuracy on valid YAMLs:

```
- Valid Deployment still predicts all keys correctly
- Valid Service still predicts all keys correctly
- Valid Pod still predicts all keys correctly
- Valid ConfigMap still predicts all keys correctly
- Valid StatefulSet still predicts all keys correctly
```

## What This Does NOT Change

- Still key-only masking, no value prediction
- No changes to the transformer encoder or attention mechanism
- No changes to the tree bias (future enhancement)
- Training data and vocabulary building process unchanged (except adding kind vocab)
