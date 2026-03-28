# YAML-BERT: Attention on Kubernetes Structured Data

## Goal

Learn the **empirical patterns and conventions** of real-world Kubernetes/OpenShift YAML manifests — the things the OpenAPI specification leaves out.

**What this is NOT:** The K8s OpenAPI spec already defines valid keys, their hierarchy, value types, enum constraints, and required fields. Re-learning that by deep learning is pointless. A schema validator already does it perfectly.

**What this IS:** Real-world K8s configs follow patterns that no spec captures:
- **Value distributions** — `replicas` is an int per spec, but 3 is common and 9999 is suspicious
- **Cross-field correlations** — if `resources.requests.memory: 2Gi`, then `limits.memory` is almost always ≥ that
- **Cross-resource relationships** — a Service's `spec.selector` must match a Deployment's `metadata.labels`, but no single resource's schema encodes that
- **Practitioner conventions** — if you have a `livenessProbe`, you almost always have a `readinessProbe`; the spec says both are optional

The model should learn the statistical regularities of how practitioners actually use Kubernetes, not what valid Kubernetes is.

**The key technical challenge:** make the attention mechanism learn that the same key (e.g., `spec`, `replicas`) has different meaning depending on its position in the YAML tree.

## Core Idea

Standard transformers use positional encoding to represent position in a sequence. YAML is a tree, not a sequence. Replace sequential positional encoding with **tree-aware positional encoding** that captures depth, parent-child relationships, and node type.

Train using masked node prediction (like BERT's masked language model) — mask out a YAML key or value, and the model predicts it from context.

## Architecture

```
YAML Document
    |
    v
YAML Parser (PyYAML) → tree of (key, value, depth, parent_path) nodes
    |
    v
Linearize tree → flat sequence of nodes
    |
    v
Token Embedding + Tree Positional Encoding
    |
    v
Transformer Encoder (multi-head self-attention, N layers)
    |
    v
Masked node prediction head
```

## Phases

### Phase 1: Data Collection & YAML Tokenizer

**Data sources:**
- Dump from a real cluster: `kubectl get <resource> -A -o yaml`
- Public Helm charts (Artifact Hub has thousands)
- Kubernetes documentation examples
- Start with 3-4 resource types: Deployment, Service, ConfigMap, Namespace

**Custom tokenizer:**
- Train a WordLevel or BPE tokenizer on YAML content only
- Vocabulary should include: Kubernetes keywords (`apiVersion`, `metadata`, `spec`, `containers`, etc.), common values (`ClusterIP`, `NodePort`, `Always`, `IfNotPresent`), structural tokens (for depth, node type)
- Add special tokens: `[MASK]`, `[PAD]`, `[CLS]`, `[SEP]`, `[KEY]`, `[VALUE]`, `[LIST_ITEM]`

**YAML to sequence conversion:**
- Parse YAML into a tree
- Linearize using DFS traversal
- Each node becomes a token with metadata: (token, node_type, depth, sibling_index, parent_token)

Example — a Deployment snippet:
```yaml
apiVersion: apps/v1
kind: Deployment
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
```

Linearized:
```
(apiVersion, KEY, depth=0)  (apps/v1, VALUE, depth=0)
(kind, KEY, depth=0)        (Deployment, VALUE, depth=0)
(spec, KEY, depth=0)
(replicas, KEY, depth=1)    (3, VALUE, depth=1)
(template, KEY, depth=1)
(spec, KEY, depth=2)
(containers, KEY, depth=3)
(name, KEY, depth=4)        (nginx, VALUE, depth=4)
(image, KEY, depth=4)       (nginx:1.21, VALUE, depth=4)
```

Note: `spec` at depth 0 and `spec` at depth 2 have the same token but different tree positions.

### Phase 2: Tree Positional Encoding

This is the novel part. Options to explore:

**Option A: Learnable depth + sibling embeddings**
```
position_encoding = depth_embedding(depth) + sibling_embedding(sibling_idx) + node_type_embedding(KEY/VALUE/LIST)
```

**Option B: Path hashing**
- Encode the full path from root to node (e.g., hash of "spec > template > spec > containers")
- Use as a learnable embedding lookup

**Option C: Relative tree distance in attention**
- Modify the attention score to include relative tree distance between two nodes
- Similar to how relative positional encoding works for sequences, but on a tree

Start with Option A — simplest to implement and debug.

### Phase 3: Training (Masked Node Prediction)

- Randomly mask 15% of tokens (same as BERT)
- Model predicts the masked token from surrounding context
- Loss: CrossEntropyLoss on masked positions only

**What success looks like:**
- Given a Deployment with `spec.replicas` masked, model predicts a reasonable number
- Given `spec.containers[].imagePullPolicy` masked, model predicts `Always` or `IfNotPresent`
- The embedding for `spec` at depth 0 (Deployment spec) should be different from `spec` at depth 2 (Pod spec) — measure with cosine similarity
- The embedding for `replicas` under `spec` (desired state, user intent) should be different from `replicas` under `status` (runtime observation, cluster-reported actual state) — same token, completely different semantics

### Phase 4: Scale to All Resource Types

- Expand data collection to all K8s resource types: Nodes, Roles, ClusterRoles, PersistentVolumes, Ingress, NetworkPolicy, etc.
- Include OpenShift-specific resources: Routes, DeploymentConfigs, BuildConfigs
- Retrain on the full corpus

### Phase 5 (Stretch): Downstream Tasks

Once the encoder is trained, use it for:
- **Anomaly detection:** encode all manifests, cluster them, flag outliers
- **Missing field prediction:** given a partial YAML, suggest what's missing
- **Cross-resource understanding:** does a Service selector match any Deployment's labels?

## Key Questions to Answer During Implementation

1. How to handle YAML values that are themselves structured (e.g., label selectors, resource quantities like `500Mi`)?
2. Should `kind` be a special token that conditions everything else, or just another node?
3. How to handle lists (containers is a list of objects)?
4. What is the right vocabulary size for a YAML-only tokenizer?
5. How much data is enough? Hundreds of manifests? Thousands?

## Tech Stack

- Python, PyTorch
- PyYAML for parsing
- HuggingFace tokenizers for custom tokenizer
- kubectl / Kubernetes API for data collection
- Repo: `llms-study/yaml-bert` (or similar)

## References

- BERT paper: "Pre-training of Deep Bidirectional Transformers" (Devlin et al.)
- Tree-Transformer: "Tree Transformer: Integrating Tree Structures into Self-Attention" (Wang et al., 2019)
- Code-BERT and similar code understanding models for inspiration on structured data
