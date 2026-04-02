# YAML-BERT v2: Kind Embedding — Conclusions

> **Historical document.** This describes the v1 -> v2 transition. The current model (v4/v5) removed kind_embedding from the input and instead uses hybrid bigram/trigram prediction targets. See [architecture.md](architecture.md) for the current design.

## What We Built

v2 added `kind_embedding` — a document-level positional component that tells every node "you're inside a Deployment" or "you're inside a Service." This was added because v1 couldn't distinguish kind-specific structures (e.g., `replicas` is valid in Deployment spec but not in Pod spec).

## What Worked

### Kind embedding partially fixes anomaly detection

v1 detected 0/8 kind-specific structural errors. v2 (epoch 5) detected 4/8:

| Test | v1 | v2 |
|------|:---:|:---:|
| template in Pod spec | FAIL | **PASS** |
| spec in ConfigMap | FAIL | **PASS** |
| spec in Secret | FAIL | **PASS** |
| schedule in Deployment | FAIL | **PASS** |
| replicas in Pod spec | FAIL | FAIL |
| containers in Deployment spec | FAIL | FAIL |
| serviceName in Deployment | FAIL | FAIL |
| replicas in Service spec | FAIL | FAIL |

### All v1 capabilities preserved

v2 passes all 22 capabilities that v1 passed. Adding kind embedding didn't hurt anything.

### Kind embedding is backward compatible

v1 checkpoints load and run in v2 code without issues.

## What Did Not Work

### 4 anomaly cases still fail

`replicas` under `spec` and `containers` under `spec` are so strongly associated with `parent_key=spec` from training data across many resource types that the kind signal cannot override it.

### Tree positional information degrades through layers

Probing analysis revealed the core problem:

| Property | Embedding Input | After Layer 5 | Loss |
|----------|:-:|:-:|:-:|
| depth | 87% | 67% | -20% |
| node_type | 95% | 79% | -16% |
| parent_key | 79% | 63% | -16% |

The tree positional encoding we designed is being washed out by the transformer layers. By the final layer, the model has lost ~20% of the structural information we gave it.

### Per-head specialization is weak

No individual attention head strongly specializes in any tree property. The best per-head accuracy for parent_key is 4.5% — essentially nothing. Information is distributed across heads rather than concentrated.

## Root Cause

**The model only optimizes key prediction loss.** It preserves tree structure only insofar as it helps predict keys. When the model finds shortcuts that don't need explicit tree structure (e.g., co-occurrence patterns), it discards the structural information.

Adding more positional components (kind, parent_key, etc.) at the input doesn't help if the transformer overwrites them by layer 5.

## Key Insight

The model learns what the loss rewards. If we want the model to preserve tree structure, we must **add auxiliary losses that explicitly reward structural preservation** — not just inject more information at the input.

## Next Step: Auxiliary Losses (v3)

```
total_loss = key_prediction_loss
           + α * depth_prediction_loss
           + β * parent_key_prediction_loss
           + γ * kind_prediction_loss
```

At each layer's output, add a prediction head that asks "do you still know the depth/parent/kind?" If the model can't answer, it gets penalized. This forces the transformer to maintain tree structure throughout all layers.

The probing classifiers we built for analysis become the auxiliary loss heads — we already have the infrastructure.

## Metrics Summary

| Metric | v1 (epoch 15) | v2 (epoch 5) |
|--------|:-:|:-:|
| Test cases passing | 68/77 | 71/77 |
| Capabilities fully passing | 22/24 | 22/24 |
| Kind-specific rejection | 0/8 | 4/8 |
| Training loss | 0.092 | 0.143 |
| Parameters | 7.36M | 7.38M |
