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
| **Parent key** | Immediate ancestor (contextual position) | parent="spec" vs parent="status" |

The fundamental difference: **sequential position is 1D, tree position is multi-dimensional.**

## Tree Positional Encoding: The Formula

We encode each dimension separately and sum:

```
TPE(node) = depth_emb(depth) + sibling_emb(sibling_index) + type_emb(node_type) + parent_emb(parent_key)
```

Each component is a learned embedding vector of the same dimension (e.g., 128). The full input to the transformer for each node is:

```
input(node) = token_emb(token) + TPE(node)
```

Where `token_emb` comes from either the key vocabulary or value vocabulary depending on node type.

## Mathematical Properties

### 1. Uniqueness

Every distinct combination of (depth, sibling, type, parent) produces a distinct vector, provided the component embeddings are linearly independent. Learned embeddings naturally become linearly independent during training — gradient descent pushes them apart to reduce loss.

For example, these two nodes get distinct encodings:

```
replicas under spec:   depth_emb(1) + sibling_emb(0) + type_emb(KEY) + parent_emb("spec")
replicas under status: depth_emb(1) + sibling_emb(0) + type_emb(KEY) + parent_emb("status")
```

They share three components but differ in `parent_emb`. The resulting vectors are distinct.

### 2. Distance-Sensitivity (Tree Proximity)

The dot product between two nodes' encodings reflects their structural similarity:

```
TPE(node_i) . TPE(node_j) = sum of dot products across all component pairs
```

**Siblings** (same parent, same depth, same type) share three of four components:

```
TPE(name at containers.0) = depth_emb(1) + sibling_emb(0) + type_emb(LIST_KEY) + parent_emb("containers")
TPE(image at containers.0) = depth_emb(1) + sibling_emb(1) + type_emb(LIST_KEY) + parent_emb("containers")
```

These share depth, type, and parent — they differ only in sibling. Their dot product is high.

**Distant nodes** (different subtrees) differ in parent and possibly depth:

```
TPE(name under metadata) = depth_emb(1) + sibling_emb(0) + type_emb(KEY) + parent_emb("metadata")
TPE(name under containers.1) = depth_emb(1) + sibling_emb(0) + type_emb(LIST_KEY) + parent_emb("containers")
```

These differ in type and parent. Their dot product is lower.

**This property emerges naturally from the additive structure**: more shared components means higher similarity, which means stronger attention between structurally related nodes.

### 3. Decomposability in Attention

The attention score between two nodes is:

```
score(i, j) = (W_Q . x_i)^T . (W_K . x_j)
```

Where `x = token_emb + TPE`. Since TPE is a sum of components, this dot product decomposes into **cross-term interactions**:

```
score ~ (depth_i x depth_j) + (depth_i x parent_j) + (parent_i x parent_j)
      + (sibling_i x sibling_j) + (type_i x type_j) + (token_i x parent_j) + ...
```

The W_Q and W_K matrices learn which cross-terms matter. Different attention heads can specialize:

- **A "parent-child" head** could learn to weight the (token_i x parent_j) cross-term — "attend to nodes whose token matches my parent key"
- **A "sibling" head** could weight (parent_i x parent_j) — "attend to nodes that share my parent"
- **A "depth-aware" head** could weight (depth_i x depth_j) — "attend to nodes at similar depth"

This decomposability means the model doesn't need separate mechanisms for different tree relationships — the multi-head attention with additive positional encoding can learn them all through the same linear algebra.

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

The model must learn that "two levels apart" is a consistent relationship across different absolute depths. With only ~10 depth levels and d_model dimensions, this is easily learnable — but it is learned, not built in.

We could recover this property by using sinusoidal encoding for the depth dimension specifically:

```
depth_component(d, 2i)   = sin(d / 10000^(2i/d_depth))
depth_component(d, 2i+1) = cos(d / 10000^(2i/d_depth))
```

This would give "attend to nodes k levels up/down" as a built-in linear operation. For our scale (small depth range, sufficient training data), this optimization is unnecessary — but it would be theoretically cleaner.

## Comparison Table

| Property | Sequential (Sinusoidal) | Tree (Learned Additive) |
|----------|------------------------|------------------------|
| Position space | 1D integer | Multi-dimensional (depth, sibling, type, parent) |
| Encoding method | Deterministic sine/cosine | Learned embedding per component, summed |
| Uniqueness | Yes (by construction) | Yes (learned; linearly independent embeddings) |
| Distance-sensitivity | Yes (dot product decreases with distance) | Yes (more shared components = higher dot product) |
| Relative position | Built-in via rotation matrices | Learned by attention heads from cross-terms |
| Extrapolation | Yes (works for unseen positions) | Limited (unseen depths/parents map to [UNK]) |
| Structural relationships | "k positions away" = rotation | "same parent", "sibling", "parent-child" = learned via cross-terms |

## The One-Sentence Summary

Sequential positional encoding answers "where am I in the sequence?" with one number. Tree positional encoding answers "where am I in the tree?" with multiple coordinates — depth, sibling order, role, and ancestry — each capturing one axis of structural position, summed into a single vector that the attention mechanism can decompose into structural relationships.

## Why This Matters for Kubernetes YAML

Consider this real example:

```yaml
spec:
  replicas: 3         # desired replica count (user intent)
status:
  replicas: 2         # actual replica count (cluster state)
```

Both `replicas` keys have the same token embedding. Without tree positional encoding, the model cannot distinguish them structurally — it would have to rely entirely on surrounding context.

With tree positional encoding, the difference is explicit:

```
TPE(spec.replicas)   = depth_emb(1) + sibling_emb(0) + type_emb(KEY) + parent_emb("spec")
TPE(status.replicas) = depth_emb(1) + sibling_emb(0) + type_emb(KEY) + parent_emb("status")
```

The parent_emb component directly encodes that one `replicas` lives under `spec` and the other under `status`. The model doesn't need to figure this out from context — it's told.

This is the core value proposition: **tree positional encoding gives the transformer explicit structural information that would otherwise have to be learned implicitly from attention patterns over linearized sequences.**
