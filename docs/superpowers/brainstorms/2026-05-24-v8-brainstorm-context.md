# v8 — Problem Statement for Brainstorming

## The data

YAML manifests for Kubernetes-style systems:

- **Structure:** tree (nested dicts and lists with scalar leaves).
- **Vocabulary:** small bounded set — ~1000 unique keys, ~80 resource "kinds" (Pod, Deployment, Service, …), each key may appear under many parents.
- **Examples:**
  ```yaml
  apiVersion: apps/v1
  kind: Deployment
  metadata:
    name: web
    labels: {app: nginx}
  spec:
    replicas: 3
    selector: {matchLabels: {app: nginx}}
    template:
      metadata: {labels: {app: nginx}}
      spec:
        containers:
        - name: nginx
          image: nginx:1.25
          ports: [{containerPort: 80}]
  ```
- **Corpus size:** ~276K documents available for self-supervised pretraining.
- **No labeled paraphrases.** No naturally-paired "equivalent but differently-worded" documents.
- **No outcome labels.** We have the YAMLs themselves, that's it (for now).

## Architecture family

The model is an **encoder-only transformer** (BERT-style), not a decoder or encoder-decoder. Pretraining objective is **masked language modeling**: hide some token positions, predict them from surrounding context.

A novel piece (already proven useful — keep): **tree positional encoding.** Instead of sequence-index positional encoding, each linearized YAML node is given embeddings for its `depth` in the tree, its `sibling_index` among siblings of the same parent, and its `node_type` (key vs value, list-key vs list-value). These sum into the input vector alongside the token embedding. The encoder is thus structurally aware — it knows where each token sits in the tree, not just where it sits in the linear sequence.

Brainstorm should stay within this family:
- Encoder-only, not generative
- MLM-style pretraining (mask + predict from context)
- Tree positional encoding retained
- Hyperparameters (d_model, num_layers, num_heads) tunable; architecture family is not

Open within the family: output heads, auxiliary objectives, training schedule, choice of what to mask, how to aggregate to document level, etc.

## What we want to compute

Two outputs, both useful, both potentially expensive:

### 1. Per-position token predictions

Given a YAML, for any position in it, predict what key (or value) belongs there. Used to surface missing/expected fields and to validate user-written manifests.

The key vocabulary is small (~1000 keys), but the *meaning* of a key depends on where it sits — `name` under `metadata` is different from `name` under `containers[0]` is different from `name` under `containers[0].ports[0]`. The prediction must be sensitive to position in the tree (not just token identity).

### 2. Per-document vector representations

Given a YAML, produce a fixed-dimensional vector that captures the document's identity and content.

Used downstream for:
- Retrieval at cluster scale (find resources similar to a query)
- Clustering and outlier detection
- Drift detection (compare same resource over time)
- Context compression for LLM agents (send a short vector instead of a long YAML)

**Our current chosen approach for producing this vector** is bottom-up aggregation along the tree:

```
For leaf nodes: node_vec = encoder hidden state
For internal nodes with children c1..cn:
    node_vec(v) = combine(node_vec(v's tokens), node_vec(c1), ..., node_vec(cn))
doc_vec = node_vec(root)
```

`combine` is some function (mean, attention, Tree-LSTM cell, etc. — also an open design choice).

This choice gives us, as a byproduct, a vector per subtree at every internal node — useful for fine-grained retrieval (e.g. "find similar livenessProbe configurations" not just "find similar Deployments").

We are open to alternatives if there's a better way to produce a document vector that doesn't rely on the tree-aggregation pattern. Possible alternatives include a learned aggregator token (CLS-style), various pooling schemes, etc. — but the brainstorm should weigh them against the natural fit of tree aggregation given that our data IS a tree.

## What's been ruled out (with evidence)

- **Atomic-only token prediction** ("predict just the key at this position"): produces representations that don't discriminate well between resource kinds. Cross-kind hidden-state similarity becomes 0.84-0.92, meaning the model treats a Pod's `metadata.name` essentially the same as a Deployment's `metadata.name`. Insufficient signal for kind-aware downstream tasks.
- **Auxiliary kind-classification head from unmasked input:** trivially solvable through residual connections (loss crashes to ~0 in a few batches without learning anything useful). Any unmasked-property auxiliary loss has this leak.
- **Contrastive learning with augmentation-based positive pairs:** YAML has no natural paraphrases. Trivial augmentations (sibling reorder) collapse to identity under tree-positional-encoding. Content-changing augmentations (drop a label, rename a key) lose document identity.

## Open design questions

1. **How to encode position-sensitive token prediction without exploding the output vocabulary?** Per-position prediction can either use a flat key vocab (~1000 classes, but loses positional context) or compound targets like `(kind, parent, child)` (28K+ classes, expensive). Is there a third way?

2. **What's the right `combine` function for our bottom-up tree aggregation?** Mean is too crude (every child equally important). Attention seems appealing (siblings have different relevance to document identity). Tree-LSTM is more expressive but costly. Alternatively — is there a fundamentally better approach to producing the document vector than tree aggregation?

3. **What pretraining objective(s) produce useful representations at both token and document levels?** MLM works for token level. What at the document level — given that we've ruled out contrastive-with-augmentations and unmasked-property-prediction-with-residual-leak?

4. **How should token-level and document-level signals interact during training?** Joint? Staged? Multi-task with weights? Or completely decoupled (train one, use it for the other downstream)?

5. **How to evaluate?** What concrete benchmarks would tell us a v8 design is better than just-pool-the-existing-v7-encoder?

## Non-goals (explicit)

- Beating frontier LLMs at general code/text understanding. (Not the goal.)
- Achieving perfect prediction. Calibrated, useful predictions are enough.
- Solving any specific downstream domain at this stage. We want a solid general-purpose YAML model first.

## What we'd like out of brainstorming

Approaches to:
- Avoid both the residual leak and the contrastive-augmentation trap.
- Produce both token-level predictions AND document/subtree vectors.
- Keep the vocabulary tractable.
- Be evaluable on benchmarks we can construct without labeled data.

Pointers to relevant literature, especially for tree-structured / hierarchical-data models.

Critiques and failure modes of proposed approaches.
