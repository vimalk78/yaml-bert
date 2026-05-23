# Inference Pipeline: Missing-Field Suggestion

How `suggest_missing_fields()` turns a YAML manifest into a ranked list
of suggested missing keys. Complements `architecture.md` (training-time
mechanics) and `suggest-demo.md` (user-facing output).

This document is also the structural blueprint for the upcoming
visualizations in the Gradio app — each step below is a potential
visualization point.

## Overview

Given an input YAML and a trained YAML-BERT model, predict what keys
are *expected* to be present at each parent level but aren't.

```
Input YAML  ──►  Linearize  ──►  Probe each parent  ──►  Aggregate & rank
                                  (one forward pass
                                   per parent)
```

The whole pipeline runs **entirely at inference** — no fine-tuning. The
model was trained on masked LM (predict masked keys); we exploit the
same machinery by inserting fake `[MASK]` tokens at structural positions
we want to interrogate.

## Phase 1 — Setup (once per document)

### 1a. Linearize the YAML

`YamlLinearizer.linearize()` walks the YAML tree depth-first, producing
a flat list of `YamlNode`s. Each node carries:

| Field | Source |
|---|---|
| `token` | the key string or value string |
| `node_type` | KEY / VALUE / LIST_KEY / LIST_VALUE |
| `depth` | tree depth (root = 0) |
| `sibling_index` | position among siblings of the same parent |
| `parent_path` | dotted path from root |

For example, a small Pod linearizes to 13 nodes:

```
idx  token       type        depth  sibling  parent_path
─────────────────────────────────────────────────────────────────────────
 0   apiVersion  KEY         0      0        ""
 1   v1          VALUE       0      0        "apiVersion"
 2   kind        KEY         0      1        ""
 3   Pod         VALUE       0      1        "kind"
 4   metadata    KEY         0      2        ""
 5   name        KEY         1      0        "metadata"
 6   app         VALUE       1      0        "metadata.name"
 7   spec        KEY         0      3        ""
 8   containers  KEY         1      0        "spec"
 9   name        LIST_KEY    2      0        "spec.containers.0"
10   web         LIST_VALUE  2      0        "spec.containers.0.name"
11   image       LIST_KEY    2      1        "spec.containers.0"
12   nginx       LIST_VALUE  2      1        "spec.containers.0.image"
```

### 1b. Encode to tensors

Each node becomes a tuple of integers via vocab lookups:

```python
token_ids  = [vocab.encode_key(n.token) if n is KEY else vocab.encode_value(n.token) ...]
node_types = [{KEY:0, VALUE:1, LIST_KEY:2, LIST_VALUE:3}[n.node_type] ...]
depths     = [min(n.depth, 15) ...]        # clamped to depth_embedding size
siblings   = [min(n.sibling_index, 31) ...]
```

All four arrays have length = number of nodes (13 in our example).

### 1c. Group keys by parent_path

```python
keys_by_parent = {
    "":                                {apiVersion, kind, metadata, spec},
    "metadata":                        {name},
    "spec":                            {containers},
    "spec.containers.0":               {name, image},
}
```

This becomes the set of probing locations. We will issue one forward
pass per entry.

## Phase 2 — Probe each parent (per-parent forward pass)

For each `parent_path`:

### 2a. Construct a fake `[MASK]` node

The probe asks: "what additional key would be expected at this parent?"
We simulate that by appending a fake `[MASK]` as the next sibling.

```python
last_pos  = positions[-1]          # idx of last existing key at this parent
last_node = nodes[last_pos]
fake_mask = YamlNode(
    token       = "[MASK]",
    node_type   = last_node.node_type,           # KEY or LIST_KEY
    depth       = last_node.depth,
    sibling_index = last_node.sibling_index + 1, # "next sibling"
    parent_path = last_node.parent_path,
)
```

### 2b. Splice it into the sequence

We walk forward from `last_pos + 1` to find where the last existing key's
subtree ends, then insert the fake `[MASK]` there:

```python
insert_pos = last_pos + 1
while insert_pos < len(nodes) and nodes[insert_pos].depth > last_node.depth:
    insert_pos += 1   # skip past last_node's subtree

# splice the fake mask's four encoded values into the four arrays
fake_token_ids = token_ids[:insert_pos] + [mask_id] + token_ids[insert_pos:]
# ... and the same for node_types, depths, siblings
```

**Note on sequence position.** Self-attention is permutation-equivariant
and tree-PE encodes structure via `(depth, sibling, parent_path)` — not
sequence index. So *any* valid splice position would produce the same
prediction at the [MASK] location. The walking logic is cosmetic
alignment with the training distribution.

### 2c. Single forward pass

```python
simple_logits, kind_logits = model(token_ids, node_types, depths, siblings)
# simple_logits.shape = (1, N+1, ~4785)
# kind_logits.shape   = (1, N+1, ~278)
```

The encoder runs over **all** positions; logits are emitted at every
position. We will only use the position where we inserted `[MASK]`.

### 2d. Route to a head

```python
if depth == 0:
    head = simple   # root keys (apiVersion, kind, metadata, spec)
elif depth == 1 and parent_key in {apiVersion, kind, metadata}:
    head = simple   # universal-parent children (metadata.name, …)
elif depth == 1 and parent_key not in {apiVersion, kind, metadata}:
    head = kind     # kind-specific (Pod::spec::replicas, Deployment::spec::template, …)
else:  # depth >= 2
    head = simple   # deep bigrams (containers::image, resources::limits, …)
```

### 2e. Kind-conditioning (when using kind_head)

We already know the document's kind from `kind: Pod` at depth 0. Mask
out kind_head logits for any target whose kind prefix doesn't match:

```python
mask = full_negative_infinity_array
for target_str, idx in kind_target_vocab.items():
    if target_str.startswith(f"{kind}::"):
        mask[idx] = 0          # keep this target
# also keep special tokens accessible
for idx in special_tokens.values():
    mask[idx] = 0

masked_logits = kind_logits[0, mask_pos] + mask
```

Bayes interpretation: we're computing `P(target | context, kind)` by
applying a known constraint (kind = "Pod"). Probability mass
redistributes onto the surviving Pod-prefixed targets via softmax
renormalization. Sharper, no wrong-kind noise.

### 2f. Softmax → top-K candidates

```python
probs = softmax(masked_logits)        # sums to 1 over kind_target_vocab
topk  = probs.topk(top_k + 5)         # tensor of top (k+5) (idx, prob) pairs
```

### 2g. Decode & per-candidate filter

For each of the (k+5) candidates:

```python
target_str = id_to_kind[idx]            # e.g., "Pod::spec::restartPolicy"
parts      = target_str.split("::")     # ["Pod", "spec", "restartPolicy"]
key_name   = parts[-1]                  # "restartPolicy"
predicted_parent = parts[-2]            # "spec"

# Reject if:
#   predicted_parent != actual_parent  → wrong-level (recorded as diagnostic)
#   key_name is a special token         → skip
#   key_name == actual_parent           → self-reference
#   key_name is a root-level key suggested at deep position → skip

predicted[key_name] = prob              # surviving candidates kept
```

After this, `predicted_keys_by_parent[parent_path]` looks like:

```python
{
    "restartPolicy":       0.45,
    "serviceAccountName":  0.22,
    "nodeName":            0.12,
    "volumes":             0.08,
    "initContainers":      0.05,
    # ... a few more under threshold ...
}
```

## Phase 3 — Aggregate & rank

After all parents have been probed:

```python
suggestions = []
for parent_path, predicted in predicted_keys_by_parent.items():
    existing = keys_by_parent[parent_path]            # already in YAML
    for key, conf in predicted.items():
        if (key not in existing                        # not already present
            and key not in _CLUSTER_MANAGED_KEYS       # not cluster-set (status, uid, …)
            and conf >= threshold):                    # passes user-set threshold
            suggestions.append({parent_path, key, conf})

suggestions.sort(by=-conf)
```

`_CLUSTER_MANAGED_KEYS = {status, creationTimestamp, generation,
resourceVersion, selfLink, uid, managedFields}` — fields the API server
writes, not the user.

## End-to-end pipeline at a glance

```
For each unique parent_path in the document:
  ┌──── per-parent loop ────────────────────────────────────────┐
  │  insert fake [MASK] as next sibling                          │
  │  forward pass → (simple_logits, kind_logits) at every pos    │
  │  pick head based on (depth, parent_key)                      │
  │  if kind_head: mask non-matching-kind targets                │
  │  softmax → top-K candidates                                  │
  │  decode + filter (wrong-parent / special / self-ref / root)  │
  │  store surviving (key, confidence) pairs                     │
  └──────────────────────────────────────────────────────────────┘

Aggregate across parents:
  drop already-present keys, cluster-managed keys, below-threshold
  sort by confidence descending
  return ranked list of (parent_path, missing_key, confidence)
```

## Optimization opportunity (not yet implemented)

The encoder emits logits at *every* position from a single forward pass.
We currently issue one forward pass per parent (~10 for a typical doc).
We could instead splice ALL fake `[MASK]`s into a single sequence and
run one forward pass — read out predictions at each mask position
simultaneously. ~10× faster. Predictions would be near-identical
because BERT-style masked predictions are largely independent across
positions.

## Visualization opportunities

Each phase above is a natural visualization point:

| Phase | Visualization |
|---|---|
| 1a. Linearization | YAML tree side-by-side with linearized node table |
| 1c. Parent grouping | List of all parent_paths and their existing children |
| 2a-b. MASK insertion | Show the sequence before and after splicing |
| 2c. Forward pass | (internal; nothing user-visible) |
| 2d. Head routing | Per parent, show which head was used and why |
| **2e. Kind-conditioning** | **Bar chart of softmax before vs after kind mask — the "aha" plot** |
| 2f. Top-K | Bar chart of top predictions per parent, sorted by probability |
| 2g. Filtering | Strikethroughs on candidates that got dropped, with reason |
| 3. Aggregation | Final ranked list, grouped by parent |

The most pedagogically rich plots are (2e), (2f), and (2g) — they show
the model's reasoning in a way the current text output hides.

## See also

- `architecture.md` — training-time mechanics, model structure
- `suggest-demo.md` — user-facing output and confidence interpretation
- `tree-positional-encoding-explained.md` — why tree-PE makes the
  permutation-equivariance argument work
- `yaml_bert/suggest.py` — the implementation
