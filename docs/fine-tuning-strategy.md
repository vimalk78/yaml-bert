# Fine-Tuning Strategy: From Structure Learning to Best Practice Suggestions

## The Problem

The pre-trained YAML-BERT model learned **what usually appears** in Kubernetes YAMLs, not **what should appear**. This leads to:

- Suggesting `capabilities.add` when the intent is `drop: ALL` (weakening security)
- Suggesting `externalIPs` on Services (common in training data, rarely needed)
- Suggesting `generateName` on pods (statistical artifact from controller-created pods)
- Missing suggestions for `livenessProbe`/`readinessProbe` (not always present in training data, but should be)

The model has strong structural understanding (93/93 capability tests, 0.42 cross-kind similarity) but no concept of "good practice" vs "common practice." Fine-tuning bridges this gap.

## Background: Pre-Training vs Fine-Tuning

### What Pre-Training Gave Us

- Tree positional encoding that understands YAML hierarchy
- Token embeddings that know key-value semantics
- Attention patterns that capture parent-child, sibling, and cross-branch relationships
- Kind-aware representations (50.7% kind probe accuracy, learned not leaked)
- Parent-aware representations (51.7% parent probe accuracy, learned not leaked)

### What Pre-Training Cannot Give Us

- Policy: "containers SHOULD have resource limits" vs "containers sometimes have resource limits"
- Intent: "drop ALL capabilities is deliberate" vs "drop ALL capabilities is incomplete"
- Priority: "missing readinessProbe is more important than missing annotations"
- Version awareness: "task v2 requires param X that v1 didn't"

## Fine-Tuning Approaches

### Approach 1: Weighted Curriculum Learning (Pre-Training Extension)

Continue pre-training but mix in curated best-practice examples at higher weight.

**How it works:**

```
Training batch composition:
  - 70% regular data (276K HuggingFace YAMLs) — structure learning
  - 30% curated best-practice YAMLs — policy learning

Weighting:
  - Regular examples: loss weight = 1.0
  - Best-practice examples: loss weight = 3.0

  Effective influence: best-practice data contributes
  30% × 3.0 = 90% of gradient vs regular 70% × 1.0 = 70%
```

**Training procedure:**

1. Start from the pre-trained checkpoint (epoch 15)
2. Build a mixed dataset: regular + curated
3. Use a `WeightedRandomSampler` to oversample best-practice examples
4. Apply higher loss weight to best-practice examples
5. Train for 3-5 more epochs with lower learning rate (1/10th of pre-training LR)

**The curated dataset needs:**
- Well-configured Deployments with: resources (requests + limits), liveness/readiness probes, security context, pod disruption budgets
- Well-configured Pods with: capabilities drop ALL (without add), read-only root filesystem, non-root user
- Well-configured Services with: proper selectors, no unnecessary externalIPs
- Well-configured StatefulSets with: volumeClaimTemplates, proper serviceName
- For each kind: 50-100 examples covering different configurations

**Pros:**
- Simple to implement — same masked key prediction task, just different data mix
- Preserves all pre-training knowledge
- No new model architecture needed

**Cons:**
- The prediction objective is still "predict masked key" — it learns what's common in the weighted data, not directly "what's missing"
- May need many curated examples to overcome the 276K regular examples
- Hard to encode negative examples ("do NOT suggest add under drop: ALL")

### Approach 2: Binary Field Presence Classifier (New Head)

Add a new prediction head that classifies: "should field X be present at this position — yes or no?"

**Architecture:**

```
Frozen pre-trained encoder
         ↓
    hidden states (256-dim per position)
         ↓
    New linear head: Linear(256, 2)  →  [should_be_present, should_not_be_present]
```

**Training data construction:**

1. Start with curated best-practice YAMLs (the "should" set)
2. For positive examples: every field present in a best-practice YAML is labeled "should be present"
3. For negative examples: deliberately strip fields and label those positions as "should be present" (the model must learn to detect the absence)
4. For hard negatives: add fields that should NOT be present (e.g., `capabilities.add` when `drop: ALL` exists) and label them "should not be present"

```
Example training pair:

Best-practice Deployment:
  spec:
    replicas: 3
    strategy:            ← label: should_be_present (if masked)
      type: RollingUpdate
    template:
      spec:
        containers:
        - resources:     ← label: should_be_present (if masked)
            limits: ...
            requests: ...
          livenessProbe: ← label: should_be_present (if masked)
          readinessProbe: ← label: should_be_present (if masked)

Stripped version (training input):
  spec:
    replicas: 3
    template:
      spec:
        containers:
        - resources:
            limits: ...
          [MASK]         ← target: "should_be_present" (requests is missing)
          [MASK]         ← target: "should_be_present" (livenessProbe is missing)
```

**Training procedure:**

1. Load pre-trained encoder (freeze or use very low LR)
2. Add new `Linear(256, 2)` head
3. Train only the new head (or head + last 1-2 encoder layers) for 10-20 epochs
4. Loss: binary cross-entropy with class weights to handle imbalance

**Pros:**
- Directly optimizes for the downstream task ("what's missing")
- Can encode "should NOT be present" (hard negatives)
- Small amount of curated data needed (encoder already understands structure)
- Fast to train — only one linear layer

**Cons:**
- Requires curated training data with explicit "should/shouldn't" labels
- Binary (yes/no) doesn't tell you WHAT should be present, just that something is missing
- Separate from the existing suggest pipeline

### Approach 3: Contrastive Fine-Tuning (Good vs Bad YAMLs)

Train the encoder to produce different representations for well-configured vs poorly-configured YAMLs.

**How it works:**

```
Pairs:
  (good_deployment_with_probes, good_deployment_with_probes)  → similar (positive pair)
  (good_deployment_with_probes, bad_deployment_without_probes) → dissimilar (negative pair)

Loss: supervised contrastive loss
  - Pull together: well-configured resources of the same kind
  - Push apart: well-configured vs poorly-configured
```

**Training data:**

For each resource kind:
- **Anchor**: well-configured example (has probes, resources, security context)
- **Positive**: another well-configured example (different names/values, same best practices)
- **Negative**: stripped version missing key best-practice fields

**Procedure:**

1. Load pre-trained encoder
2. Add attention pooling layer (learned query → document embedding)
3. Train with supervised contrastive loss
4. At inference: compare user's YAML embedding to known good/bad centroids

**Pros:**
- Produces document-level "quality score"
- Can cluster by configuration quality
- Natural ranking: "this YAML is 80% similar to best practice"

**Cons:**
- Requires pairs of good/bad examples
- Document-level, not field-level (tells you something is wrong, not what)
- Needs the attention pooling layer we discussed (additional component)

### Approach 4: Combined (Recommended)

Use Approach 1 (weighted curriculum) for continued pre-training, then Approach 2 (binary classifier) for the suggest tool.

**Phase 2a: Weighted curriculum learning (2-3 epochs)**
- Continue from epoch 15 checkpoint
- Mix in curated best-practice examples at 3x weight
- Same masked key prediction objective
- Result: model now biases toward best-practice patterns in its predictions

**Phase 2b: Binary field presence head (10-20 epochs, head only)**
- Freeze the encoder from Phase 2a
- Add a new binary classification head
- Train on "should this field be present" examples
- Result: direct signal for the suggest tool

**Phase 2c: Contrastive pooling (optional)**
- Add attention pooling on top of Phase 2a encoder
- Train with good/bad YAML pairs
- Result: document-level quality scoring

Each phase builds on the previous. Phase 2a improves the base representations. Phase 2b adds a task-specific head. Phase 2c adds document-level understanding.

## Curated Dataset Requirements

### What We Need

For each major resource kind, curated examples representing best practices:

| Kind | Key Best Practices | Examples Needed |
|------|-------------------|-----------------|
| Deployment | resources (requests+limits), liveness/readiness probes, security context, strategy, PDB | 50-100 |
| Pod | same as container-level, plus serviceAccountName, securityContext at pod level | 50-100 |
| StatefulSet | volumeClaimTemplates, serviceName, updateStrategy | 30-50 |
| DaemonSet | updateStrategy, tolerations for node types | 30-50 |
| CronJob | concurrencyPolicy, successfulJobsHistoryLimit, backoffLimit | 30-50 |
| Job | backoffLimit, restartPolicy: Never, activeDeadlineSeconds | 30-50 |
| Service | selector, proper port naming, sessionAffinity where needed | 30-50 |
| Ingress | TLS, proper path types, backend service references | 30-50 |
| NetworkPolicy | default deny + explicit allow rules | 20-30 |
| RBAC | least-privilege roles, proper binding structure | 20-30 |

**Total: ~400-600 curated examples**

### Where to Get Them

1. **Internal clusters**: dump from well-maintained production namespaces (sanitized)
2. **Official K8s docs**: example YAMLs from kubernetes.io are well-written
3. **Security benchmarks**: CIS Kubernetes Benchmark provides security-hardened examples
4. **Manually curated**: take existing training data, add missing best-practice fields
5. **Helm chart defaults**: popular charts (nginx-ingress, prometheus, cert-manager) represent community best practices

### Hard Negatives (What NOT to Suggest)

Equally important — examples where a field's absence is intentional:

| Pattern | Why It's Intentional |
|---------|---------------------|
| `capabilities: {drop: [ALL]}` without `add` | Security hardening — no capabilities needed |
| Service without `selector` | Headless service or ExternalName |
| Pod without `livenessProbe` | Short-lived jobs, init containers |
| Deployment without `strategy` | Default RollingUpdate is fine |
| ConfigMap without `labels` | System-generated, not user-managed |

These hard negatives prevent the model from blindly suggesting fields that aren't needed.

## Implementation Details

### Weighted Sampler for Curriculum Learning

```python
from torch.utils.data import WeightedRandomSampler

# Assign weights: 1.0 for regular, 3.0 for curated
weights = []
for i in range(len(dataset)):
    if dataset.is_curated[i]:
        weights.append(3.0)
    else:
        weights.append(1.0)

sampler = WeightedRandomSampler(weights, num_samples=len(dataset))
dataloader = DataLoader(dataset, batch_size=24, sampler=sampler)
```

### Binary Classifier Head

```python
class FieldPresenceHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.classifier = nn.Linear(d_model, 2)

    def forward(self, hidden_states: torch.Tensor,
                mask_positions: torch.Tensor) -> torch.Tensor:
        # Extract hidden states at masked positions only
        masked_states = hidden_states[mask_positions]
        return self.classifier(masked_states)
```

### Training Data Generator for Binary Classifier

```python
def generate_training_pairs(good_yaml: str, fields_to_strip: list[str]):
    """Generate (input, label) pairs from a best-practice YAML.

    1. Parse the YAML
    2. For each field in fields_to_strip:
       - Remove it from the YAML
       - Add a [MASK] at that position
       - Label: should_be_present = 1
    3. For fields that ARE present:
       - Mask them
       - Label: should_be_present = 1 (they should stay)
    4. For hard negatives:
       - Insert a field that shouldn't be there
       - Label: should_be_present = 0
    """
    ...
```

### Learning Rate Schedule

```
Phase 2a (curriculum):
  Encoder LR: 1e-5 (1/10th of pre-training 1e-4)
  Prediction heads LR: 1e-4
  Epochs: 3-5
  Warmup: 500 steps

Phase 2b (binary head):
  Encoder LR: 0 (frozen) or 1e-6 (barely tuned)
  Binary head LR: 1e-3
  Epochs: 10-20
  No warmup needed (small head)
```

## Evaluation

### Metrics for Fine-Tuned Suggest Tool

1. **Precision**: what fraction of suggestions are actually good practice?
   - Target: >80% (up from current ~28% on cluster data)

2. **Recall**: what fraction of missing best practices does it catch?
   - Target: >60% (currently low — misses probes, security context)

3. **False positive rate on complete YAMLs**: suggestions on already-correct configs?
   - Target: <5% (currently good — near 0% on production deployments)

4. **Hard negative accuracy**: does it correctly NOT suggest `capabilities.add` under `drop: ALL`?
   - Target: >95%

### Test Sets

1. **Curated test set**: held-out best-practice YAMLs (not used in training)
2. **Deliberately broken YAMLs**: good configs with fields stripped — model should catch
3. **Intentionally minimal YAMLs**: configs where minimalism is correct (Jobs, headless Services)
4. **Real cluster data**: the 6686 YAMLs from the OpenShift cluster

## Summary

| Phase | Objective | Data | New Architecture | Training Time |
|-------|-----------|------|------------------|---------------|
| 1 (done) | Structure learning | 276K HuggingFace | Pre-trained encoder | 15 epochs (~15h) |
| 2a | Best practice bias | 276K + 400-600 curated | Same (weighted loss) | 3-5 epochs (~5h) |
| 2b | Field presence detection | 400-600 curated (augmented) | + Binary head | 10-20 epochs (~1h) |
| 2c | Document quality scoring | Good/bad YAML pairs | + Attention pooling | 5-10 epochs (~2h) |

Phase 2a and 2b together give us a suggest tool that knows the difference between "common" and "correct."
