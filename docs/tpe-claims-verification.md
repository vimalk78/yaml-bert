# TPE Claims Verification

Empirical verification of the three mathematical claims made in
[tree-positional-encoding-explained.md](tree-positional-encoding-explained.md):
uniqueness, distance-sensitivity, and decomposability in attention.

The numbers below were measured on the v5 model. The TPE architecture
— `depth + sibling + node_type` summed at the input — is unchanged in
v9: same embedding-table shapes, same additive composition. The
structural claims still hold. (v9 added byte-level BPE on top of the
input, which changes how *tokens* are represented but not how
*positions* are encoded.)

Reproduce with `scripts/test_tpe_claims.py <checkpoint> --vocab <vocab>`.

## Claim 1: Uniqueness

Every distinct combination of (depth, sibling, type) produces a
distinct vector, provided the component embedding tables are linearly
independent.

```
Claim 1: UNIQUENESS
  320 TPE vectors tested (10 depths x 8 siblings x 4 types)
  51,040 pairs checked — all distinct (no cosine > 0.99)
  Depth embeddings:     rank 10/10 (full rank, linearly independent)
  Sibling embeddings:   rank 8/8
  Node type embeddings: rank 4/4
  PASS
```

## Claim 2: Distance-Sensitivity

The dot product between two nodes' encodings reflects their structural
similarity — more shared `(depth, sibling, type)` components → higher
cosine.

```
Claim 2: DISTANCE-SENSITIVITY
  3 shared components: cosine = 1.0000
  2 shared components: cosine = 0.6386 avg
  1 shared component:  cosine = 0.2871 avg
  0 shared components: cosine = -0.1566
  PASS (monotonic decrease: 3 > 2 > 1 > 0)
```

Caveat: *which* two components are shared doesn't matter much —
siblings are not more similar to each other than arbitrary
2-shared-component pairs.

## Claim 3: Decomposability in Attention

The attention dot product over additive TPE decomposes into cross-term
interactions. Different heads can specialize on different positional
dimensions.

```
Claim 3: DECOMPOSABILITY IN ATTENTION
  28/48 attention heads are depth-specialized (>2x bias)
  1/48 heads are type-specialized
  Max depth bias: 14.50x (Layer 3 Head 4)
  PASS (structural specialization observed)
```

The model relies heavily on **depth** for structural attention. Sibling
and node-type specialization are much weaker — most of the structural
work happens through depth-aware heads.
