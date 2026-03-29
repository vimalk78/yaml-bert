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

This is a breaking architecture change. The new model has an additional embedding table, so existing checkpoints cannot be loaded. Retraining is required.

## Impact on Existing Tests

All capability tests should continue to pass (the model gets strictly more information). The anomaly detection should improve significantly for kind-specific errors.

## What This Does NOT Change

- Still key-only masking, no value prediction
- No changes to the transformer encoder or attention mechanism
- No changes to the tree bias (future enhancement)
- Training data and vocabulary building process unchanged (except adding kind vocab)

## Open Questions

1. Should `kind` be extracted during linearization (added to `YamlNode`) or during dataset creation (extracted from the raw document)?
   - Recommendation: during dataset creation — keeps `YamlNode` unchanged, kind is a document-level property not a per-node property.

2. Should documents without `kind` (unlikely but possible) get a special token or `[UNK]`?
   - Recommendation: `[UNK]` — treat as generic, model learns from context.
