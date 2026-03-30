# YAML-BERT v4: Complete Architecture Specification

## What This Is

A transformer-based model that learns structural patterns from Kubernetes YAML manifests using tree-aware positional encoding and hybrid prediction targets.

The model predicts masked keys in linearized YAML trees. Keys under `metadata` (universal structure) are predicted as simple bigrams (`metadata::name`). Keys under `spec` (kind-specific structure) are predicted as trigrams (`Deployment::spec::replicas`). This follows the actual Kubernetes object model where TypeMeta/ObjectMeta are shared across all resources while Spec/Status are type-specific.

## Input Representation

A YAML document is linearized into a flat sequence of nodes via DFS traversal. Each node is a `(token, node_type, depth, sibling_index, parent_path)` tuple.

Example:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
```

Linearized:
```
(apiVersion, KEY, depth=0, sibling=0, parent_path="")
(apps/v1, VALUE, depth=0, sibling=0, parent_path="apiVersion")
(kind, KEY, depth=0, sibling=1, parent_path="")
(Deployment, VALUE, depth=0, sibling=1, parent_path="kind")
(metadata, KEY, depth=0, sibling=2, parent_path="")
(name, KEY, depth=1, sibling=0, parent_path="metadata")
(web, VALUE, depth=1, sibling=0, parent_path="metadata.name")
(spec, KEY, depth=0, sibling=3, parent_path="")
(replicas, KEY, depth=1, sibling=0, parent_path="spec")
(3, VALUE, depth=1, sibling=0, parent_path="spec.replicas")
```

## Embedding Layer

4 embedding tables, all projecting to `d_model` dimensions, summed into one input vector per node:

| Table | Size | Index | Role |
|---|---|---|---|
| `token_embedding` (keys) | key_vocab_size × d_model | Key token ID from key vocabulary | "I am this key" — e.g., `spec`, `replicas`, `name` |
| `token_embedding` (values) | value_vocab_size × d_model | Value token ID from value vocabulary | "I am this value" — e.g., `nginx`, `3`, `Always` |
| `depth_embedding` | max_depth × d_model | Depth integer (0, 1, 2, ...) | "I am at this depth in the tree" |
| `sibling_embedding` | max_sibling × d_model | Sibling index integer (0, 1, 2, ...) | "I am the Nth child of my parent" |
| `node_type_embedding` | 4 × d_model | NodeType enum (KEY=0, VALUE=1, LIST_KEY=2, LIST_VALUE=3) | "I am a key / value / list key / list value" |

Token embedding is routed by node_type: KEY/LIST_KEY use the key embedding table, VALUE/LIST_VALUE use the value embedding table.

### What Is NOT in the Embedding

- **No `parent_key_embedding`** — parent context is captured by the hybrid prediction target. The model learns parent awareness from the bigram targets.
- **No `kind_embedding`** — kind conditioning is captured by the trigram prediction targets for spec children. The model learns kind awareness from the training objective.

### Formula

```
input(node) = LayerNorm(token_emb(token) + depth_emb(depth) + sibling_emb(sibling) + type_emb(node_type))
```

## Transformer Encoder

Standard PyTorch TransformerEncoder. No modifications to the attention mechanism.

| Parameter | Value |
|---|---|
| d_model | 256 |
| num_layers | 6 |
| num_heads | 8 |
| d_ff | 1024 (4 × d_model) |
| batch_first | True |

Each layer: multi-head self-attention → add & LayerNorm → feed-forward → add & LayerNorm.

## Prediction Heads

Two separate linear prediction heads on the final encoder output:

### Simple Head

Predicts universal structure — root keys and everything under metadata.

```python
self.simple_head: nn.Linear = nn.Linear(d_model, simple_vocab_size)
```

Target vocabulary: bigram tokens like `metadata::name`, `containers::image`, `env::value`, plus unigrams for root keys (`apiVersion`, `kind`, `metadata`, `spec`).

- Vocabulary size: ~1,700 at min_freq=100
- Parameters: 256 × 1,700 = ~435K

### Kind-Specific Head

Predicts kind-dependent structure — first-level children under any kind-specific root key (everything except `apiVersion`, `kind`, `metadata`).

```python
self.kind_head: nn.Linear = nn.Linear(d_model, kind_vocab_size)
```

Target vocabulary: trigram tokens like `Deployment::spec::replicas`, `Service::spec::ports`, `ConfigMap::data::DB_HOST`, `ClusterRole::rules::apiGroups`.

- Vocabulary size: ~300-500 at min_freq=100 (includes all kind-specific root keys, not just spec)
- Parameters: 256 × 500 = ~128K

### Which Head Predicts Which Node

Determined by tree position — a static rule, not learned:

```python
# Root keys that are universal (TypeMeta + ObjectMeta)
UNIVERSAL_ROOT_KEYS = {"apiVersion", "kind", "metadata"}

def get_head(node: YamlNode) -> str:
    if node.depth == 0:
        return "simple"                              # root keys themselves
    parent_key = extract_parent_key(node.parent_path)
    if node.depth == 1 and parent_key not in UNIVERSAL_ROOT_KEYS:
        return "kind_specific"                       # first-level under kind-specific root
    return "simple"                                  # everything else
```

| Tree position | Head | Target example |
|---|---|---|
| Root keys (depth=0) | Simple | `"metadata"` (unigram) |
| Under metadata | Simple | `"metadata::name"` (bigram) |
| First level under spec | Kind-specific | `"Deployment::spec::replicas"` (trigram) |
| First level under data | Kind-specific | `"ConfigMap::data::DB_HOST"` (trigram) |
| First level under rules | Kind-specific | `"ClusterRole::rules::apiGroups"` (trigram) |
| First level under subjects | Kind-specific | `"RoleBinding::subjects::kind"` (trigram) |
| First level under roleRef | Kind-specific | `"RoleBinding::roleRef::apiGroup"` (trigram) |
| First level under webhooks | Kind-specific | `"ValidatingWebhookConfiguration::webhooks::name"` (trigram) |
| Deeper under any root key | Simple | `"containers::image"` (bigram) |

### The Rule

All root keys except `apiVersion`, `kind`, and `metadata` are kind-specific. Their first-level children use trigram targets with the kind prefix. This covers:
- `spec` (Deployment, Service, Pod, StatefulSet, ...)
- `status` (runtime state)
- `data` (ConfigMap, Secret)
- `rules` (ClusterRole, Ingress, NetworkPolicy)
- `subjects` (RoleBinding, ClusterRoleBinding)
- `roleRef` (RoleBinding, ClusterRoleBinding)
- `webhooks` (ValidatingWebhookConfiguration, MutatingWebhookConfiguration)
- Any other kind-specific root key

## Hybrid Prediction Targets

Inspired by n-gram language models. The target for each masked key includes context proportional to how kind-specific the position is.

### Unigram (root keys)
```
apiVersion → "apiVersion"
kind → "kind"
metadata → "metadata"
spec → "spec"
```

### Bigram (universal structure — parent::key)

Used for children under `metadata` (ObjectMeta) and for deeper nested nodes everywhere:
```
metadata.name → "metadata::name"
metadata.labels → "metadata::labels"
metadata.namespace → "metadata::namespace"
containers.image → "containers::image"
containers.name → "containers::name"
ports.containerPort → "ports::containerPort"
env.name → "env::name"
selector.matchLabels → "selector::matchLabels"
```

### Trigram (kind-specific — kind::root_key::key)

Used for first-level children under any kind-specific root key:
```
Deployment::spec::replicas
Deployment::spec::selector
Deployment::spec::template
Deployment::spec::strategy
Service::spec::ports
Service::spec::selector
Service::spec::type
Pod::spec::containers
Pod::spec::volumes
StatefulSet::spec::serviceName
ConfigMap::data::DB_HOST
Secret::data::password
ClusterRole::rules::apiGroups
ClusterRole::rules::resources
ClusterRole::rules::verbs
RoleBinding::subjects::kind
RoleBinding::roleRef::apiGroup
Ingress::spec::rules
NetworkPolicy::spec::podSelector
```

The `::` separator avoids ambiguity with dots in K8s key names like `app.kubernetes.io/name`.

### Why This Scheme Works

Follows the Kubernetes object model. Every K8s resource embeds:

```go
type AnyResource struct {
    metav1.TypeMeta   `json:",inline"`      // → unigram targets (universal)
    metav1.ObjectMeta `json:"metadata"`     // → bigram targets (universal)
    Spec   PowerMonitorSpec   `json:"spec"` // → trigram targets (kind-specific)
    Status PowerMonitorStatus `json:"status"` // → trigram targets (kind-specific)
}
```

- `TypeMeta` (kind, apiVersion) — same for every resource → unigram
- `ObjectMeta` (name, namespace, labels) — same for every resource → bigram
- `Spec` — different per kind → trigram with kind prefix
- Template contents (`spec.template.spec.*`) — shared across workload controllers (Deployment, DaemonSet, StatefulSet, Job) → bigram (universal)

## Loss Computation

```python
def compute_loss(self, simple_logits, simple_labels, kind_logits, kind_labels):
    simple_loss = CrossEntropyLoss(ignore_index=-100)(
        simple_logits.view(-1, simple_logits.size(-1)),
        simple_labels.view(-1),
    )
    kind_loss = CrossEntropyLoss(ignore_index=-100)(
        kind_logits.view(-1, kind_logits.size(-1)),
        kind_labels.view(-1),
    )
    return simple_loss + kind_loss
```

No weighting needed — both losses use CrossEntropyLoss with `ignore_index=-100`. Nodes handled by the simple head get -100 in kind_labels (ignored), and vice versa.

## Masking Strategy

Same as v1-v3:
- Only KEY and LIST_KEY nodes are candidates for masking
- VALUE and LIST_VALUE are never masked — they serve as context
- 15% of eligible key nodes are masked per document
- Of masked nodes: 80% replaced with [MASK], 10% random key, 10% unchanged

## Vocabulary

### Token Vocabularies (input)

| Vocabulary | Size | Purpose |
|---|---|---|
| Key vocabulary | ~1,664 (at min_freq=100) | Token IDs for key nodes |
| Value vocabulary | ~5,193 (at min_freq=100) | Token IDs for value nodes |
| Special tokens | 3 ([PAD]=0, [UNK]=1, [MASK]=2) | Masking and padding |

### Target Vocabularies (output)

| Vocabulary | Size | Purpose |
|---|---|---|
| Simple targets | ~1,700 | Unigrams + bigrams for simple head |
| Kind-specific targets | ~300-500 | Trigrams for kind-specific head (all kind-specific root keys, not just spec) |

Built from 276,520 documents in the substratusai/the-stack-yaml-k8s dataset.

## Dataset

Each training example produces two label tensors:

```python
{
    "token_ids": tensor,          # input token IDs (some masked)
    "node_types": tensor,         # KEY/VALUE/LIST_KEY/LIST_VALUE
    "depths": tensor,             # tree depth per node
    "sibling_indices": tensor,    # sibling position per node
    "simple_labels": tensor,      # bigram/unigram target IDs (-100 for kind-specific positions)
    "kind_labels": tensor,        # trigram target IDs (-100 for simple positions)
}
```

## Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| d_model | 256 | Sufficient for ~276K document corpus |
| num_layers | 6 | Deep enough for structural pattern learning |
| num_heads | 8 | 32 dims per head |
| d_ff | 1024 | 4 × d_model |
| max_depth | 16 | Covers K8s YAML nesting |
| max_sibling | 32 | Covers widest sibling counts |
| mask_prob | 0.15 | Standard BERT masking rate |
| lr | 1e-4 | Standard for transformer pre-training |
| batch_size | 32 | Fits in 4GB VRAM |
| num_epochs | 15 | Convergence expected by epoch 10 |
| max_seq_len | 512 | Headroom for large manifests |

## Model Size

| Component | Parameters |
|---|---|
| Key token embedding | 1,664 × 256 = 426K |
| Value token embedding | 5,193 × 256 = 1,329K |
| Depth embedding | 16 × 256 = 4K |
| Sibling embedding | 32 × 256 = 8K |
| Node type embedding | 4 × 256 = 1K |
| LayerNorm (embedding) | 512 |
| Transformer (6 layers) | ~4,800K |
| Simple prediction head | 1,700 × 256 = 435K |
| Kind-specific prediction head | 500 × 256 = 128K |
| **Total** | **~7.1M** |

## VRAM Estimate

| Component | Size |
|---|---|
| Model weights (fp32) | 28 MB |
| Optimizer states (AdamW, 2×) | 56 MB |
| Gradients | 28 MB |
| Simple logits (32 × 512 × 1700) | 107 MB |
| Kind logits (32 × 512 × 500) | 31 MB |
| Attention matrices (6 layers) | 300 MB |
| Other activations | ~200 MB |
| **Total** | **~740 MB** |

Fits comfortably in 4GB VRAM.

## Expected Outcomes

1. **Key prediction accuracy**: ≥95% on simple targets (same as v1). Kind-specific accuracy depends on target vocabulary coverage.

2. **Discriminative document embeddings**: hidden states for `spec.replicas` in a Deployment will differ from `spec.replicas` in a Pod because the trigram targets are different. Mean-pooled document embeddings should show lower cross-kind cosine similarity (target: <0.7 vs current 0.89).

3. **Kind-specific rejection**: `replicas` under spec in a Pod should have low confidence because `Pod::spec::replicas` is rare/absent in the trigram vocabulary.

4. **Generalization**: universal structure (metadata, container fields) generalizes to unseen CRDs. Kind-specific structure (spec first-level children) is learned per-kind.

5. **Convention detection**: the suggest tool works unchanged — runner-up predictions still indicate missing fields.

## Files Changed

| File | Change |
|---|---|
| `yaml_bert/embedding.py` | Remove kind_emb and parent_key_emb (4 tables instead of 6) |
| `yaml_bert/model.py` | Two prediction heads (simple + kind-specific), updated forward and compute_loss |
| `yaml_bert/vocab.py` | Build simple and kind-specific target vocabularies |
| `yaml_bert/dataset.py` | Generate simple_labels and kind_labels tensors |
| `yaml_bert/trainer.py` | Compute hybrid loss from both heads |
| `yaml_bert/config.py` | No new hyperparameters |
| `scripts/train_hf.py` | Updated for new vocab and label generation |
