# Inference Pipeline: Missing-Field Suggestion

How `suggest_missing_fields()` turns a YAML manifest into a ranked list
of suggested missing keys. Complements `architecture.md` (training-time
mechanics). For live user-facing examples, see the
[HF Space](https://huggingface.co/spaces/vimalk78/yaml-bert).

## Overview

Given an input YAML and a trained YAML-BERT model, predict what keys
are *expected* to be present at each parent level but aren't.

```
Input YAML  ──►  Linearize  ──►  Probe each parent  ──►  Aggregate & rank
                                  (one forward pass
                                   per parent)
```

The whole pipeline runs **entirely at inference** — no fine-tuning. The
model was trained on masked KEY prediction (whole-word MLM); we exploit
the same machinery by inserting a fake `[MASK]` at each structural
position we want to interrogate.

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

### 1b. Encode to subword tensors

The v9 vocabulary uses byte-level BPE. Each logical node's token is
encoded as a **list** of subword ids — keys typically resolve to 1
subword, values to 1-N subwords:

```python
# Each logical node expands to its subword span. logical_id tracks which
# logical node each subword belongs to (needed for pooling later).
sub_token_ids = []
sub_logical_ids = []
sub_node_types = []
sub_depths = []
sub_siblings = []

for logical_idx, node in enumerate(nodes):
    is_value = node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE)
    pieces = vocab.encode_token(node.token, is_value=is_value)  # list[int]
    for sid in pieces:
        sub_token_ids.append(sid)
        sub_logical_ids.append(logical_idx)
        sub_node_types.append(node_type_to_int[node.node_type])
        sub_depths.append(min(node.depth, 15))
        sub_siblings.append(min(node.sibling_index, 31))
```

All five arrays have the same length — the **subword sequence length**
(typically 1.5-2× the logical-node count). Building the batch via
`YamlBertDataset` + `collate_fn` precomputes the tree tensors
(`parent_of_tensor`, `top_level_key_mask`, `edges_by_depth`,
`parents_by_depth`) the aggregator needs.

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
nodes.insert(insert_pos, fake_mask)
```

**Subword expansion.** One logical `[MASK]` node expands to a single
`[MASK]` subword id (the special token isn't BPE-split), so the dataset
collate replicates the tree metadata exactly once. (For real
multi-subword KEYs, whole-word masking applies — all subwords of one
logical KEY become `[MASK]` together. Probes only insert single-subword
masks.)

**Note on sequence position.** Self-attention is permutation-equivariant
and tree-PE encodes structure via `(depth, sibling, parent_path)` — not
sequence index. So *any* valid splice position would produce the same
prediction at the [MASK] location. The walking logic is cosmetic
alignment with the training distribution.

### 2c. Single forward pass

```python
logits, doc_vec = model(
    token_ids=batch["token_ids"],
    node_types=batch["node_types"],
    depths=batch["depths"],
    sibling_indices=batch["sibling_indices"],
    batch_info=batch["batch_info"],
    padding_mask=batch["padding_mask"],
    logical_ids=batch["logical_ids"],
    n_logical_per_doc=batch["n_logical_per_doc"],
    parent_of_tensor=batch["parent_of_tensor"],
    top_level_key_mask=batch["top_level_key_mask"],
    edges_by_depth=batch["edges_by_depth"],
    parents_by_depth=batch["parents_by_depth"],
)
# logits.shape  = (1, L_max, 11080)   <- LOGICAL-level, atomic_target_vocab
# doc_vec.shape = (1, 256)
```

Important shape note for v9: **logits are at the logical-node level, not
the subword level.** The encoder runs over all subword positions, but
the model's `_pool_subwords` mean-pools subwords of each logical node
back into one hidden vector per logical, and the Key Head outputs over
those logical positions. We read out logits at the **logical position
of the inserted [MASK]**, not at any subword index.

The Key Head input at each logical position is
`[h_logical ; doc_vec ; s_parent]` — local pooled hidden, whole-document
vector, and immediate parent's subtree vector. The kind context that
older versions encoded with a separate `kind_head` now reaches the
prediction implicitly through `doc_vec` (the document's broad signature)
and `s_parent` (the parent subtree). One head, no routing.

### 2d. Softmax → top-K candidates

```python
mask_logical_pos = ...  # the logical-position index of the fake [MASK]
probs = softmax(logits[0, mask_logical_pos])    # (11080,)
topk_probs, topk_ids = probs.topk(top_k + 5)
```

No per-kind masking is needed at inference. The `atomic_target_vocab` is
a flat set of atomic key strings (e.g., `containers`, `replicas`,
`restartPolicy`), not kind-prefixed entries. Doc_vec + s_parent already
sharpen the distribution toward kind-appropriate keys.

### 2e. Decode & per-candidate filter

For each of the (k+5) candidates:

```python
target_str = id_to_atomic[idx]          # e.g., "restartPolicy"
                                        # (no Kind:: prefix anymore)

# Reject if:
#   key_name is a special token         → skip
#   key_name == parent_key              → self-reference
#   key_name is a root-level key suggested at deep position → skip
#   key_name already exists at this parent → skip

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
  ┌──── per-parent loop ──────────────────────────────────────────┐
  │  insert fake [MASK] as next sibling                            │
  │  forward pass → (logits, doc_vec) at every logical position    │
  │  read logits at the masked logical position                    │
  │  softmax → top-K candidates                                    │
  │  decode + filter (special / self-ref / root / already-present) │
  │  store surviving (key, confidence) pairs                       │
  └────────────────────────────────────────────────────────────────┘

Aggregate across parents:
  drop already-present keys, cluster-managed keys, below-threshold
  sort by confidence descending
  return ranked list of (parent_path, missing_key, confidence)
```

## Optimization opportunity (not yet implemented)

The encoder emits logits at *every* logical position from a single
forward pass. We currently issue one forward pass per parent (~10 for a
typical doc). We could instead splice ALL fake `[MASK]`s into a single
sequence and run one forward pass — read out predictions at each mask
position simultaneously. ~10× faster. Predictions would be
near-identical because BERT-style masked predictions are largely
independent across positions.

## Visualization opportunities

Each phase above is a natural visualization point:

| Phase | Visualization |
|---|---|
| 1a. Linearization | YAML tree side-by-side with linearized node table |
| 1b. Subword encoding | Show the BPE expansion: each logical → its subword span |
| 1c. Parent grouping | List of all parent_paths and their existing children |
| 2a-b. MASK insertion | Show the sequence before and after splicing |
| 2c. Forward pass | Per-position attention rollout (where did this prediction look?) |
| 2d. Top-K | Bar chart of top predictions per parent, sorted by probability |
| 2e. Filtering | Strikethroughs on candidates that got dropped, with reason |
| 3. Aggregation | Final ranked list, grouped by parent |

The most pedagogically rich plots are (2d) and (2e) — they show the
model's distribution before and after filtering. A diagnostic worth
adding: alongside the top-K bar chart, render `[h_logical ; doc_vec ;
s_parent]` as three separate heatmaps to surface which signal is
driving each prediction.

## See also

- `architecture.md` — training-time mechanics, model structure
- [HF Space](https://huggingface.co/spaces/vimalk78/yaml-bert) — live missing-field suggester
- `tree-positional-encoding-explained.md` — why tree-PE makes the
  permutation-equivariance argument work
- `key-value-design-rationale.md` — why values aren't predicted
- `yaml_bert/suggest.py` — the implementation
