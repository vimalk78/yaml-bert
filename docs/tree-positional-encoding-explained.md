# Tree Positional Encoding: From Sequences to Trees

## The Problem

Transformers process sequences. YAML is a tree. When we linearize a YAML tree into a flat sequence for the transformer, we lose structural information. Two nodes with the same token (e.g., `spec`) might appear at different positions in the tree with completely different meanings:

- `spec` at the top level of a Deployment: defines the desired state
- `spec` nested under `template`: defines the Pod specification

Standard sequential positional encoding (position 1, 2, 3, ...) would tell the model these tokens are at different *sequence* positions, but not that they're at different *tree* positions. The model would have to infer tree structure entirely from context, rather than being told it directly.

Tree positional encoding solves this by encoding *where in the tree* a node sits, not just where in the linearized sequence.

## Sequential Positional Encoding (Background)

In the original Transformer paper (Vaswani et al., 2017), positional encoding uses sine and cosine functions at different frequencies:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Where `pos` is the position in the sequence, `i` is the dimension index, and `d` is the embedding dimension.

### Why Sine and Cosine?

This choice is not arbitrary. It gives three key mathematical properties:

**1. Uniqueness.** Each position maps to a distinct vector. No two positions share the same encoding.

**2. Bounded values.** All values lie in [-1, 1], keeping the encoding numerically stable regardless of sequence length.

**3. Relative position as a linear transformation.** This is the deepest property. A shift by k positions can be expressed as a rotation matrix applied to the encoding:

```
PE(pos + k) = R_k . PE(pos)
```

This works because of the trigonometric addition identities:

```
sin(pos + k) = sin(pos)cos(k) + cos(pos)sin(k)
cos(pos + k) = cos(pos)cos(k) - sin(pos)sin(k)
```

In matrix form, for each frequency:

```
[sin(pos + k)]   [cos(k)   sin(k)] [sin(pos)]
[cos(pos + k)] = [-sin(k)  cos(k)] [cos(pos)]
```

This is a 2D rotation by angle `k`. The matrix `R_k` depends only on the offset `k`, not on the absolute position. This means the model can learn to attend to "the token k positions away" by learning a single linear transformation — the concept of relative distance is baked into the encoding's algebra.

**4. Distance-sensitivity.** The dot product between two position encodings decreases as the distance between them increases. Nearby tokens have more similar encodings than distant tokens:

```
PE(5) . PE(6) > PE(5) . PE(100)
```

This gives the model an inductive bias toward local attention.

## From Sequences to Trees

In a sequence, position is a single integer: "I am token #5."

In a tree, position is a **path from the root**: "I arrived here via root -> spec -> template -> spec -> containers -> [0] -> name."

This path has multiple dimensions of information:

| Dimension | What it captures | Example |
|-----------|-----------------|---------|
| **Depth** | How many steps from root (vertical position) | depth=3 means 3 levels deep |
| **Sibling index** | Position among siblings (horizontal position) | sibling=0 means first child |
| **Node type** | Structural role | KEY, VALUE, LIST_KEY, LIST_VALUE |

The fundamental difference: **sequential position is 1D, tree position is multi-dimensional.**

## Tree Positional Encoding: The Formula

We encode each dimension separately and sum:

```
TPE(node) = depth_emb(depth) + sibling_emb(sibling_index) + type_emb(node_type)
```

Each component is a learned embedding vector of dimension d_model (256). The full input to the transformer for each node is:

```
input(node) = token_emb(token) + TPE(node)
```

Where `token_emb` comes from either the key vocabulary or value vocabulary depending on node type.

### What Is NOT in the Input

Notably, we do **not** encode the parent key or the resource kind in the input embedding. Earlier versions of the model did — but this caused the residual connection to carry these features through all layers unchanged. Auxiliary losses asking "predict the parent" or "predict the kind" were trivially solved by reading from the residual, contributing zero learning signal.

Instead, parent and kind information appear in the **prediction target**: the model must predict `parent::key` (bigram) or `kind::parent::key` (trigram) rather than just the key. This forces the model to learn parent-child and kind-specific structure through attention, not input copying.

## Mathematical Properties

### 1. Uniqueness

Every distinct combination of (depth, sibling, type) produces a distinct vector, provided the component embeddings are linearly independent. Learned embeddings naturally become linearly independent during training — gradient descent pushes them apart to reduce loss.

> **Verified:** All 320 tested TPE vectors (10 depths x 8 siblings x 4 types) are distinct. All three embedding tables have full matrix rank. See [TPE claims verification](tpe-claims-verification.md).

For example, these two nodes get distinct encodings:

```
containers under spec.template:  depth_emb(3) + sibling_emb(0) + type_emb(KEY)
replicas under spec:             depth_emb(1) + sibling_emb(0) + type_emb(KEY)
```

They share sibling and type but differ in depth. The resulting vectors are distinct.

### 2. Distance-Sensitivity (Tree Proximity)

The dot product between two nodes' encodings reflects their structural similarity:

```
TPE(node_i) . TPE(node_j) = sum of dot products across all component pairs
```

**Siblings** (same depth, same type, different sibling index) share two of three components:

```
TPE(name at depth 4)  = depth_emb(4) + sibling_emb(0) + type_emb(KEY)
TPE(image at depth 4) = depth_emb(4) + sibling_emb(1) + type_emb(KEY)
```

**Distant nodes** (different subtrees) share fewer components:

```
TPE(root key)         = depth_emb(0) + sibling_emb(0) + type_emb(KEY)
TPE(deep nested value) = depth_emb(4) + sibling_emb(3) + type_emb(VALUE)
```

More shared components means higher dot product, which means stronger attention between structurally related nodes.

> **Verified:** Monotonic decrease in similarity as shared components decrease: 3 shared = 1.0, 2 shared = 0.64 avg, 1 shared = 0.29 avg, 0 shared = -0.16. However, *which* two components are shared doesn't matter much — siblings are not more similar than other 2-shared-component pairs. See [TPE claims verification](tpe-claims-verification.md).

### 3. Decomposability in Attention

The attention score between two nodes is:

```
score(i, j) = (W_Q . x_i)^T . (W_K . x_j)
```

Where `x = token_emb + TPE`. Since TPE is a sum of components, this dot product decomposes into **cross-term interactions**:

```
score ~ (depth_i x depth_j) + (sibling_i x sibling_j) + (type_i x type_j)
      + (token_i x depth_j) + (depth_i x token_j) + ...
```

The W_Q and W_K matrices learn which cross-terms matter. Different attention heads can specialize:

- **A "depth-aware" head** could weight (depth_i x depth_j) — "attend to nodes at similar depth"
- **A "sibling" head** could weight (sibling_i x sibling_j) — "attend to nearby siblings"
- **A "token-depth" head** could weight (token_i x depth_j) — "attend to specific tokens at specific depths"

This decomposability means the model doesn't need separate mechanisms for different tree relationships — the multi-head attention with additive positional encoding can learn them all through the same linear algebra.

> **Verified:** 28 of 48 attention heads show depth specialization (>2x bias toward same-depth nodes). Layer 3 Head 4 attends 14.5x more to same-depth nodes. Only 1 head shows type specialization — the model relies heavily on depth for structural attention. See [TPE claims verification](tpe-claims-verification.md).

### 4. What We Lose Compared to Sinusoidal

Sinusoidal sequential encoding has one property our learned tree encoding does not guarantee: **translational equivariance in depth**.

In sinusoidal:
```
PE(pos+k) - PE(pos) = constant for all pos (given fixed k)
```

The "distance" between position 3 and 5 is the same operation as between 7 and 9. This is because the shift is a rotation, and rotations compose.

In our learned depth embedding:
```
depth_emb(3) - depth_emb(1) is NOT guaranteed to equal depth_emb(5) - depth_emb(3)
```

The model must learn that "two levels apart" is a consistent relationship across different absolute depths. Empirically, we found that the learned depth embeddings are **nearly orthogonal** — the model treats each depth as an independent category rather than learning a smooth gradient. Adjacent depths (0 and 1) are no more similar than distant depths (0 and 10).

This suggests the model found categorical depth (each level is a distinct context) more useful than ordinal depth (levels form a gradient) for the prediction task. A potential improvement: initialize depth embeddings with sinusoidal encoding to provide a smooth starting point that the model can refine.

## Comparison Table

| Property | Sequential (Sinusoidal) | Tree (Learned Additive) | Verified? |
|----------|------------------------|------------------------|-----------|
| Position space | 1D integer | Multi-dimensional (depth, sibling, type) | — |
| Encoding method | Deterministic sine/cosine | Learned embedding per component, summed | — |
| Uniqueness | Yes (by construction) | Yes — full rank embeddings, 320/320 vectors distinct | PASS |
| Distance-sensitivity | Dot product decreases with distance | 3 shared=1.0, 2=0.64, 1=0.29, 0=-0.16 | PASS |
| Decomposability | Built-in via rotation matrices | 28/48 heads depth-specialized, max 14.5x bias | PASS |
| Extrapolation | Works for unseen positions | Limited — unseen depths clamped to max | — |
| Depth structure | Smooth (nearby positions similar) | Categorical — adjacent depths are orthogonal, not smooth | [Tested](../scripts/test_embedding_structure.py) |

## The One-Sentence Summary

Sequential positional encoding answers "where am I in the sequence?" with one number. Tree positional encoding answers "where am I in the tree?" with multiple coordinates — depth, sibling order, and structural role — each capturing one axis of structural position, summed into a single vector that the attention mechanism can decompose into structural relationships.

## Why This Matters for Kubernetes YAML

Consider this real example:

```yaml
spec:
  replicas: 3         # desired replica count (user intent)
status:
  replicas: 2         # actual replica count (cluster state)
```

Both `replicas` keys have the same token embedding. With tree positional encoding, the difference is partially encoded:

```
TPE(spec.replicas)   = depth_emb(1) + sibling_emb(0) + type_emb(KEY)
TPE(status.replicas) = depth_emb(1) + sibling_emb(0) + type_emb(KEY)
```

These TPE vectors are **identical** — both nodes are at the same depth, same sibling index, same type. The model must rely on **attention to context** (the unmasked `spec` and `status` keys nearby in the sequence) to distinguish them. This is intentional: parent information is encoded in the prediction target (`spec::replicas` vs `status::replicas`), forcing the model to learn contextual disambiguation through attention rather than reading a parent embedding from the residual stream.

This is the core design principle: **tree positional encoding provides structural coordinates (depth, sibling, type) that are hard to infer from context, while parent and kind information are placed in the prediction target to force the model to learn them through attention.**
