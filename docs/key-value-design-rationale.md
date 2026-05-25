# KEYs vs VALUEs: First-Class vs Second-Class — Design Rationale

This document captures the architectural decision to treat YAML KEYs and
VALUEs asymmetrically in YAML-BERT, why it's correct for the K8s manifest
domain, and the corner cases where it could bite.

## The setup

A YAML manifest linearizes into a sequence of nodes, each tagged as one of
four types:

- `KEY` — a mapping key (e.g., `apiVersion`, `spec`, `containers`)
- `VALUE` — a mapping value (e.g., `apps/v1`, `nginx`, `3`)
- `LIST_KEY` — a key inside a list item (treated like KEY for our purposes)
- `LIST_VALUE` — a value inside a list item (treated like VALUE)

KEYs define the structure of the document (the schema). VALUEs are the
content the user fills in.

## The asymmetry in v8

KEYs are first-class:
- Embedded via `key_embedding` table
- Participate in self-attention (along with VALUEs)
- **Aggregated bottom-up into `subtree_vecs` and `doc_vec`**
- **Eligible for MLM masking**
- **Predicted by the atomic Token Head**

VALUEs are second-class:
- Embedded via separate `value_embedding` table
- Participate in self-attention (along with KEYs)
- **NOT aggregated into `subtree_vecs`** — leaves in the tree, ignored by combine
- **NEVER masked** during MLM training
- **NEVER predicted** by the Token Head

Concrete code references:

```python
# yaml_bert/v8_dataset.py — MLM masking
_MASKABLE_TYPES = (NodeType.KEY, NodeType.LIST_KEY)
# VALUE positions never reach the masking branch.

# yaml_bert/aggregator.py — _forward_vectorized
# Aggregates only over edges between KEY positions (built in v8_collate_fn).
# VALUE positions exist in hidden_states but are not summed into subtree_vecs.

# yaml_bert/v8_model.py — Token Head
# atomic_logits = self.token_head([h_i ; doc_vec ; s_parent])
# Trained against atomic_labels, which are -100 (ignored) for VALUE positions.
```

## Why this is the right design for K8s YAML

The asymmetry is intentional. K8s YAML is more like a typed configuration
language than free text. The grammar lives in the keys; the values are user
payload.

### Three categories of VALUEs

1. **Payload values** (the majority).
   `name: web`, `image: nginx:1.25`, `app: frontend`, `replicas: 3`,
   environment variable values, label values, command arguments.
   - User-determined. Arbitrary content. The exact string tells us about
     ONE user's deployment, not about K8s structure.
   - Building a learned vector for `image=nginx:1.25.3` doesn't generalize
     to anything else in the K8s world.
   - Correctly treated as second-class. Their job is to be context for
     surrounding KEYs (via self-attention), nothing more.

2. **Schema values** (a small but important set).
   `kind: Pod` / `kind: Deployment` / `kind: Service`, `apiVersion: v1` /
   `apiVersion: apps/v1`, K8s enums (`type: ClusterIP`,
   `restartPolicy: Always`, `strategy.type: RollingUpdate`,
   `imagePullPolicy: IfNotPresent`).
   - These VALUEs function more like keywords than payload. Their value
     determines what other fields are valid.
   - v8 handles them implicitly: the encoder's self-attention propagates
     value information from these positions into the surrounding KEY hidden
     states. Those KEY hidden states then flow into `doc_vec` via the
     aggregator.
   - Empirical validation (v8 at 5K corpus): kind probe = 100.0%,
     apiVersion probe = 99.7%, GroupVersionKind (apiVersion+Kind combined)
     probe = 99.9%. The model has learned to encode schema values without
     directly aggregating them.

3. **Foreign-key values** (a real corner case).
   `metadata.name: web` referenced by another manifest's
   `selector.matchLabels.app: web`. `volumes[*].name: data` matched against
   `volumeMounts[*].name: data` within the same Pod.
   `serviceAccountName: backend` referencing a ServiceAccount's name.
   - These VALUEs create cross-position consistency constraints.
   - v8 cannot reason about these — not because of value second-class
     treatment, but because vanilla transformers don't enforce graph
     consistency. Even with first-class VALUEs, "this volumeMount references
     a defined volume" requires the model to check that two strings match
     at specific positions, which attention rarely does reliably.
   - This is an open problem regardless of value treatment.

## Why this design serves the embedding-model vision

If YAML-BERT is meant to produce embeddings for downstream agent pipelines
(retrieval, clustering, drift detection), second-class VALUEs are
specifically beneficial:

- Two Deployments running different apps with the same structural patterns
  (same kinds of containers, same resource limits structure, same volume
  setup) should embed CLOSE in vector space — because they ARE structurally
  similar.
- Making VALUEs first-class would inject user-specific noise (the literal
  name `web` vs `api`, the literal image `nginx` vs `httpd`) into the
  embedding, pushing structurally-identical manifests apart.
- For "find similar K8s configurations," ignoring payload values is the
  *right* thing to do. The current design naturally provides this.

## Where this design could bite

Honest catalog of cases where second-class VALUEs are a limitation:

1. **Few-shot recognition of user-specific value patterns.**
   "Find all manifests using image `mycompany/internal-tool`." The model
   has learned a generic representation of `image` keys but not specific
   value content. A user querying for specific values would need to bypass
   the embedding and use literal string matching.

2. **Image-version reasoning.**
   "Is `nginx:1.25` newer than `nginx:1.20`?" Requires understanding the
   internal structure of the version string. Subword tokenization of
   VALUEs would help here. Currently each unique image string is one
   atomic VALUE token; relationships between versions are invisible.

3. **Cross-position consistency checks.**
   See foreign-key values above. Requires reasoning the architecture
   doesn't directly support.

4. **Schema validation.**
   "Is `kind: Deployment` + `apiVersion: v1` a legal combination?" The
   model has learned co-occurrence patterns from training data but cannot
   reject invalid combinations as a hard constraint. Both v7 and v8 share
   this limitation.

## Future directions, in increasing scope

If a downstream task ever needs value semantics beyond what self-attention
provides, the following extensions are possible:

### Option A: Aggregate VALUE hidden states into parent KEY's subtree_vec

Minimal change. The aggregator's combine step:

```python
# current
subtree_vec[parent_key] = mean(own_hidden, *child_KEY_subtree_vecs)

# extended
subtree_vec[parent_key] = mean(own_hidden, *child_KEY_subtree_vecs, *immediate_VALUE_hidden)
```

Adds direct value influence to `doc_vec`. Cheap (one-line change). Could
help if probes show value-derived features are under-represented at scale.

### Option B: Mask and predict VALUEs

Bigger change. Adds VALUE positions to `_MASKABLE_TYPES`, builds a
value-prediction head, doubles the prediction problem. Forces the encoder
to learn value semantics, not just key structure. Trades training cost for
representation richness.

### Option C: Subword tokenize VALUEs

Architectural change. Replace atomic VALUE vocab with a BPE/WordPiece
tokenizer trained on VALUE strings. Lets `apps/v1`, `apps/v1beta1`,
`extensions/v1beta1` share sub-tokens, generalize across version bumps,
handle novel apiVersions for unseen CRDs. Significant work
(~1-2 weeks): tokenizer training, embedding redesign, vocab and dataset
changes.

### Option D: Schema-vs-payload distinction

Treat schema VALUEs (kind, apiVersion, known enums) as first-class while
keeping payload VALUEs second-class. Requires a curated list of "schema
keys" whose values matter structurally. More effective than B+C combined
for K8s-specific use, but K8s-specific (not portable to other YAML
domains).

## Why we're keeping the current design

For the current focus (structural understanding + general-purpose document
embedding), the asymmetric design is well-suited:

- Empirically validated: probes for kind, apiVersion, has-containers,
  has-init, has-volume-mounts, has-tolerations all clear 95% at 5K-doc
  scale. v8 at 5K already hits 112/121 capability tests vs v7 at full
  corpus 120/121.
- Conceptually clean: KEYs encode "what could be here" (schema). VALUEs
  encode "what is here" (payload). Each goes to the right place.
- Embedding-friendly: payload-blind embeddings cluster similar structures
  together, which is what retrieval/clustering use cases want.

Revisit only if downstream tasks demonstrate concrete need for value
semantics that self-attention can't provide.

## References

- `yaml_bert/v8_dataset.py` — `_MASKABLE_TYPES`, V8Dataset.__getitem__
- `yaml_bert/aggregator.py` — `_forward_vectorized`, `_forward_reference`
- `yaml_bert/v8_model.py` — V8Model.forward
- `yaml_bert/embedding.py` — separate key_embedding + value_embedding
  tables
- `docs/architecture.md` — broader architecture overview
- `docs/v8-phase1-reconstruction-results.md` — finer-grained probe results
  showing schema values are encoded in doc_vec
