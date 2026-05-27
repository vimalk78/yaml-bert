# Future Directions for YAML-BERT

Brainstorming captured from a session exploring where YAML-BERT could go
beyond the current missing-field suggester. Mix of app ideas,
architecture extensions, integration patterns, and reasoning sketches.

Not a roadmap. Pick from this when planning next work.

## App ideas

### Dropped after evaluation

- **Anomaly / typo detector.** Duplicates `kubectl --dry-run=client`,
  `kubeval`, `kubeconform`. ML adds noise where deterministic checkers
  give exact answers. Only edge value: CRDs without installed schemas,
  or "schema-valid but statistically unusual" (high false-positive).

### Promising (model-unique value)

- **Kind classifier from structure.** Strip `kind:` and `apiVersion:`,
  use `kind_head` as an implicit classifier by summing softmax mass
  per kind prefix. Zero architecture change. Useful when users paste
  partial snippets.

- **Similar-manifest search.** Encoder embeddings → fixed-dim vector
  per manifest → cosine retrieval. "Find me other Pipelines like this
  one." No retraining needed for a baseline; would benefit from attention
  pooling + proxy training (see Architecture).

- **Konflux/Tekton-specific suggester.** Same architecture as current
  app but retrained on Tekton-heavy corpus. Frontier LLMs are weak here
  because Konflux is internal/proprietary — data hasn't leaked into
  their training. Real moat.

- **PR diff explainer.** For each change in a diff, score how surprising
  it is (model assigns probability to changed fields). Flag low-prob
  changes for reviewer attention.

## Architecture extensions

### Document-level vector (in-flight: `yaml_bert/attention_pooling.py`)

Three options ranked by cost:

1. **Mean-pool**: free, baseline. Average encoder final-layer outputs.
2. **Attention-pool**: small head with `W_doc` learnable, frozen
   encoder + train on proxy task (e.g., kind classification). Hours,
   not days. Module already exists, not wired.
3. **CLS token**: full retrain, BERT-style. Probably unnecessary —
   tree-PE already gives every position structural meaning, so the
   BERT-CLS argument doesn't carry over.

Recommendation: prove mean-pool baseline first; train attention-pool
only if mean-pool outliers are noisy.

### Separate spec / status embeddings

Single vector mixes user intent and cluster reality. Separating gives:

- **Stable retrieval**: `spec_vec` only changes on edits, not on
  controller churn.
- **Drift detection**: compare query's `status_vec` to "typical
  status_vec for resources with similar spec_vec." Hard to do
  kind-agnostically with kubectl alone.
- **Health clustering**: cluster `status_vec`s across kinds — "find
  everything in the cluster that looks broken."

Optional three-way split:
- `intent_vec` (spec + user-written labels/annotations)
- `state_vec` (status only)
- `identity_vec` (name/namespace/ownerRefs — for metadata filtering,
  not similarity)

**The actual blocker is training data, not architecture.** The
architectural change is mechanically straightforward — fork the
aggregator into two paths keyed on `parent_path.startswith("status.")`,
emit two doc_vecs. The hard part is having enough status data to
produce a status_vec that's actually useful.

Our training corpus (`substratusai/the-stack-yaml-k8s`) is
GitHub-scraped YAMLs. Users commit specs to git; controllers fill in
status at runtime, never committed. Concrete signal from earlier
analysis:

- v7's bigger-boat `vocab_gap` test failed at 0/4 because status keys
  (`status.replicas`, `status.conditions`) didn't make it into the
  compound target vocab at all from the training corpus.
- v8 at full 276K atomic vocab covers them (4/4 on `vocab_gap`),
  but they're still rare — most documents in the corpus have no
  status block at all.
- Even if we separate the vectors architecturally today, the
  status_vec would be trained on a tiny fraction of the corpus and
  most documents would feed it nothing.

For a useful status_vec we need a different data source. Candidates,
in increasing cost:

1. **Kubernetes e2e test snapshots.** The k/k repo's e2e tests produce
   real `kubectl get -o yaml` output with status. Smaller corpus
   (~thousands of docs) but real status, kind-balanced.
2. **Cluster scrapes from running clusters.** Need operational access
   plus a privacy story. Largest realistic corpus; covers production
   patterns. Easier inside a single organization (Konflux) than
   public.
3. **Synthetic status via controller simulation.** Deploy spec
   manifests to a `kind` cluster, wait for controllers to converge,
   `kubectl get -o yaml`. Lots of plumbing, but generates labeled
   pairs (spec, status_after_convergence) for free.

**Recommendation:** don't pursue spec/status separation until we have
a status-rich training corpus. The architecture work is easy and can
land in days; the data work is the actual cycle. Until then, the
single doc_vec is good enough for the spec-focused use cases (which
are most of them — retrieval, similarity, structural classification).

Track this in the "Cluster collection pipeline" section below — once
that pipeline produces status-bearing data at scale, this becomes
the next architectural cycle.

### Multi-task heads

Single encoder, multiple downstream heads:
- anomaly score (continuous)
- complexity score (continuous)
- kind classifier (categorical)
- workload-type classifier (web/batch/ML/database/sidecar — latent
  property, no schema field for it)

Each head ~10K params. Train heads independently with frozen encoder.

## AI assistant integration patterns

The model's niche in a chatbot/agent architecture is **fast cluster-scale
embedding + scoring** that an LLM can call as a tool. The LLM handles
reasoning, generation, and conversation; the small model handles things
that don't scale to LLM-per-resource.

### Use cases ranked by leverage

1. **Structural RAG over cluster state.** LLM can't reason over 5K
   resources; embed them all, do k-NN, send top-K to LLM.
2. **Context compression.** Turn a 500-line Pipeline into a 50-token
   structural summary the LLM consumes cheaply.
3. **Anomaly/outlier scoring at cluster scale.** Score every resource;
   LLM explains the top 20.
4. **Field suggestion as agent tool.** Current app, wrapped as MCP/tool.
5. **Drift detection.** Embedding deltas between snapshots flag
   significant changes for LLM review.
6. **Cross-resource consistency.** Cluster similar Pipelines, suggest
   shared base templates.

### Where the model is the wrong tool

- Schema validation → `kubeconform` (deterministic, exact)
- Owner-reference traversal → K8s API
- Constraint reasoning → OPA / Kyverno (exact)
- Explanations / remediation prose → LLM

### Architecture

```
User question → LLM agent decides what to query
              → [YAML-BERT service: embed/score/suggest at scale]
              → Returns top-K relevant resources, anomaly scores, gaps
              → LLM reasons over the focused result set
              → Natural-language answer
```

**Deployment shape**: in-cluster operator preferred over centralized
SaaS — data never leaves the customer cluster. Matches Konflux's
internal-customer model.

## Cluster collection pipeline (for embedding service)

1. **Collect**: K8s informer/watch (real-time) or `kubectl get` snapshots
   (simple). RBAC ask is sensitive — cluster-wide read may not be
   palatable; namespace-scoped operator more so.
2. **Preprocess**: drop `status`, `managedFields`, `resourceVersion`,
   `uid`, `creationTimestamp`, `generation`, `ownerReferences`,
   `last-applied-configuration` annotation. Without this, embeddings
   are dominated by controller churn rather than user intent.
3. **Embed**: linearize → encoder → pool. Batch 32–128 per forward pass.
   5K resources on CPU: minutes. 50K: needs batching/GPU.
4. **Store**: small (<10K): numpy + SQLite. Larger: Qdrant / Weaviate
   / pgvector with `(cluster, namespace, kind)` index for pre-filtering.
5. **Update**: snapshot mode (cron) or informer mode (always-fresh).
   Hybrid is realistic: nightly full snapshot + watch high-churn kinds.

## Reasoning capability (what's realistic)

Important honest framing: most "ML reasoning" decomposes into
classification + retrieval + generation + scoring. Pure vector-only
reasoning that isn't reducible to one of these is rare.

### YAML-native reasoning tasks (most also doable by OPA/Kyverno)

1. Cross-resource constraint reasoning (Deployment + HPA bounds) →
   OPA does this. ML adds nothing.
2. Configuration consistency conflicts (`runAsNonRoot: true` +
   `runAsUser: 0`) → Kyverno does this.
3. Counterfactual via simulation (construct hypothetical YAML, run
   policy) → deterministic and exact.
4. Multi-step refinement (Service needs Endpoints) → linter rule.

### Where vectors uniquely add value

1. **Latent property inference** — "what kind of workload is this?" No
   schema field. Patterns in image, resources, sidecars, env vars
   provide signal. OPA can't infer; vectors can.

2. **Counterfactual probability prediction** — "if I add
   `runAsNonRoot: true`, what's the probability it breaks?" Learned
   from observed change outcomes. OPA enumerates known-failure rules
   but can't reason about unknown ones.

3. **Semantic equivalence between syntactically different manifests** —
   borderline; OPA normalization could handle simple cases.

The cleanest "needs vectors" case is (2), and it's the hardest to
realize because of data requirements.

### Counterfactual probability prediction — concrete sketch

**Data sources** (Konflux has a head start here):
- GitOps history: commit + sync success/revert
- Tekton PipelineRun outcomes: every failed run is labeled negative
  with the manifest version that ran
- Cluster audit logs + Prometheus correlation: heavy infrastructure
  ask but possible
- Manual postmortem labeling: slow but high-signal

**Don't try to learn arbitrary counterfactuals.** Learn type-specific
predictors: one classifier per common change type (e.g., "add
securityContext.runAsNonRoot", "reduce replicas", "tighten NetworkPolicy").
~500–2000 examples per (change_type × kind) cell. With 20-30 change
types, 10K–50K labels total — hard but tractable for a mature org.

**Formulation**:
```
Input:  encoder(manifest_before) + structured(change_description)
Output: P(success) via sigmoid head
Loss:   binary cross-entropy on (success/fail) labels
```

Encoder frozen or fine-tuned. Small MLP head. Standard supervised.

**Honest accuracy ceiling**: probably 70-80%, not 95%. Useful for
ranking/recommendation, not autonomous decisions. The product question
is whether 75%-accurate counterfactual prediction is useful as a
recommendation system.

**Easier fallback if too hard**: post-hoc retrieval — given a change
that happened, retrieve historically-similar (change, outcome) pairs
to show "this kind of change usually causes this kind of problem."
Retrieval over outcomes instead of prediction of outcomes.

**Timeline if pursued**: 3-6 months of focused work from "we have data"
to "usable classifier for top 5 change types." Real research project,
not a weekend hack.

**Konflux-specific framing**: "for Konflux Pipeline changes, predict if
this change will cause the next PipelineRun to fail." Narrow,
well-labeled, high-value. More tractable than the universal version.

## Konflux-specific opportunities

(Tied to [[project_konflux]] memory.)

Cross-cutting theme: Konflux is internal/proprietary, so frontier LLMs
haven't seen the data. Specialized model has a moat there that doesn't
exist for generic K8s.

- Tekton Pipeline missing-task suggester
- PipelineRun anomaly detection (vs typical PipelineRuns for this
  Pipeline)
- Similar-Pipeline retrieval (find templates within the org)
- Counterfactual prediction for Pipeline changes (uses outcome data
  Konflux already produces)
- Refactoring suggestions: cluster near-identical Pipelines, suggest
  shared ClusterTask abstractions

## V8 aggregation applications (post-v8 deployment)

The v8 architecture (validated at full 276K corpus) produces two
representation types that v7 doesn't have:

- **`doc_vec`**: one `d_model`-dim vector per document, summarizing
  whole-document structure. Empirically encodes kind/apiVersion/GVK at
  ~100% and within-kind structural features at 95-99%.
- **`subtree_vecs`**: one vector per KEY position, representing the
  subtree rooted at that key. Built bottom-up by the aggregator.

These enable applications that v7 cannot do. The ranking below is by
"best demonstrates v8's aggregation capability" — secondary by build
effort and user value. To be revisited when planning the next
Space app or pipeline mini-cycle.

### `doc_vec`-based apps

#### 1. Manifest similarity search (HF Space candidate)

User pastes a YAML. App returns top-5 structurally-similar manifests
from a pre-computed corpus (e.g., 10K K8s docs with cached `doc_vec`).

- **Showcases**: doc_vec geometry — kind probe at 100% is abstract,
  "here are 5 Pods that look like yours" is visual proof
- **Build**: 1-2 days. Pre-compute corpus doc_vecs, numpy + cosine
  index, paste-YAML → cards UI
- **User value**: "What does a canonical X look like?", template
  discovery, learning by example
- **Production angle**: API endpoint for agent pipelines doing
  retrieval-augmented K8s reasoning

#### 2. Manifest galaxy (HF Space candidate — cheapest visual)

Pre-computed t-SNE/UMAP of 5-10K manifests projected to 2D, colored
by kind. Interactive plot.

- **Showcases**: aggregation visually — each cluster proves doc_vec
  organizes meaningfully. Outliers within a cluster are striking.
  Cross-kind proximity (Pod-Deployment near each other; ConfigMap far
  from workloads) reveals what the model considers similar.
- **Build**: half-day. Pre-compute once, plot with Plotly. Static
  deploy, no inference at user time.
- **User value**: Mostly demonstrative. Conversation starter for
  "what is this model doing?"

#### 3. Manifest anomaly detection

User pastes a YAML. App computes cosine distance to nearest-k
same-kind manifests in corpus → anomaly score + the nearest neighbors
so the user sees what "normal" looks like.

- **Showcases**: within-kind variation captured by doc_vec
- **Build**: 1 day. Same infrastructure as #1 + scoring step
- **User value**: Security (rogue manifest detection), compliance
  scanning, drift detection from a baseline corpus
- **Production angle**: agent pipeline for cluster monitoring —
  "alert when a new manifest is unusual"

### `subtree_vecs`-based apps (under-explored, most differentiated)

#### 4. Subtree similarity ("find similar container specs")

User pastes a YAML manifest and clicks on a specific subtree (e.g.,
`spec.containers[0]`). App returns 5 similar subtrees from corpus.

- **Showcases**: every level of the tree has its own meaningful
  representation, not just the top. UNIQUE to v8 — v7 has no
  equivalent.
- **Build**: 2-3 days. UI: YAML viewer with clickable subtrees.
  Pre-compute subtree_vecs for corpus, store indexed by subtree path.
- **User value**: "Show me how others configured their resource
  limits / liveness probes / volume mounts." Very developer-relevant.
- **Production angle**: code-review assistant for K8s configs

#### 5. Structural diff (tree-aware)

Two YAMLs side-by-side. App aligns their subtrees via subtree_vec
cosine + highlights structurally similar vs different blocks.

- **Showcases**: tree-aware diff is novel — text-diff misses structural
  similarity (different ordering = different diff; same structure =
  same vector)
- **Build**: 3-5 days. Diff UI is non-trivial.
- **User value**: Code review for K8s configs, migration tooling,
  "what changed between these two Deployment versions?"

### Hybrid token + doc_vec apps

#### 6. Smart YAML autocomplete (v8) — replaces current Space app

Same UX as current HF Space missing-field-suggester, but using v8's
atomic prediction conditioned on `[h_i ; doc_vec ; s_parent]`. App-side
post-processing reconstructs the compound path from atomic prediction
+ position (since v8 outputs `containers` rather than
`spec.containers`).

- **Showcases**: atomic prediction with doc_vec conditioning. Subtle —
  same UX as current, but with v8's likely-better in-context accuracy
  on edge cases.
- **Build**: 1 day. Adapt the existing app to load YamlBertModel and add the
  atomic → compound reconstruction step.
- **User value**: Same as current app, potentially better. Path to
  retiring v7 from production.
- **Acceptance gate**: v8 must match or beat v7 on the capability tests
  before swapping the Space app over.

### Recommended sequence

1. **Galaxy (#2)** first — half-day, immediately visual, demonstrates
   aggregation cheaply.
2. **Similarity search (#1)** second — 1-2 days, concrete user value,
   directly uses doc_vec geometry.
3. **v8 autocomplete (#6)** to replace the current v7 app, contingent
   on capability-test parity.
4. **Subtree similarity (#4)** later — most differentiated, most useful
   for real developer workflows.

Each builds on a pre-computed embedding corpus, so the infrastructure
compounds.

### v8.1 candidate — shrink value_vocab dramatically [SUPERSEDED BY v9]

This entire section is superseded by v9 sub-tokenization (see
[v9-subword-results.md](v9-subword-results.md)). v9 took a different
path that solved the same problem more thoroughly:

- v8.1 plan: drop low-freq atomic values to shrink the table (38K → 4K)
- v9 actual: replace atomic values entirely with byte-level BPE subwords
  (one unified 8K vocab for both keys and values). Total embedding
  shrank 11.4M → 2.1M params. Model went 22.5M → 18.4M (-18%).
- v9 also fixed the `[UNK]` collision class as a side effect.

Kept here as historical record because the diagnostic insight (value
vocab is dominated by user-specific junk) was the seed for v9.

### Original v8.1 analysis (kept for context)

Empirical finding from inspecting the v8 276K vocab.json: the
`value_vocab` (38,362 entries at `value_min_freq=10`) is dominated by
user-specific junk that doesn't generalize:

- Random port numbers (`50060`, `35697`, ...)
- CRD descriptions — full English sentences from K8s API schemas
- Code blocks pasted into ConfigMap `data:` (terraform, scripts)
- App-specific names (`menu-api`, `hipster-frontend`, ...)
- CLI args (`--upstream=http://...`, `--canary-image`)

Of ~40 random samples, only 2 looked schema-relevant. The TOP of the
vocab is essential (kinds, apiVersions, K8s enums, TCP/http, common
booleans/numerics) but the long tail is wasted capacity.

`value_embedding` table is 38,362 × 256 = **9.8M params, ~44% of the
22.5M model**. Most of those params memorize user-specific strings.

Frequency distribution shows the curve is sharply concentrated:

| value_min_freq | tokens | embed params | corpus coverage |
|---|---|---|---|
| 10 (current) | 38,344 | 9.8M | 85.6% |
| 100 | 3,915 | 1.0M | 70.2% |
| 500 | 935 | 0.24M | 59.2% |
| 1000 | 491 | 0.13M | 53.7% |

Retraining v8 with `value_min_freq=100` would shrink the model by
~9M params (22.5M → ~14M) while keeping all schema-meaningful values.
Long-tail values would map to `[UNK]` — fine because v8 treats VALUEs
as second-class anyway (never aggregated into doc_vec, never predicted
by Token Head). The encoder still sees value POSITIONS via positional
embeddings + node_type — it just doesn't get a unique vector per
random container name.

Suggested v8.1 mini-cycle:
- Retrain at `value_min_freq=100` (and maybe `=500` in parallel) on
  the 276K corpus
- Run the full 4-test benchmark + 15 probes
- Acceptance gate: zero capability regression vs current v8 MLM+recon
- If passes: deploy the smaller model. Re-upload to HF Space at
  ~14M params instead of 22.5M.

Cost: ~$6 (two parallel runs) + ~1 day analysis. Big payoff if it
works: smaller model, smaller vocab.json, less embedding-memorization
of user payload, less storage / upload time / inference memory.

## v10 candidates (post-v9, 2026-05-27)

After v9 sub-tokenization shipped, several follow-up directions emerged.
None are blocking; ranked by how much they'd teach us.

### 1. Recon redesign or removal

Current recon (bag-of-keys prediction over masked subtrees) was
essentially a no-op in both v8 and v9 — loss stuck at ~0.0003 throughout
training, contributing <0.15% of gradient signal. The bag-of-keys
formulation is too easy: with 11K classes and ~30 positives per subtree,
the model wins by learning class frequencies rather than understanding
subtree structure.

Options:
- **Drop recon entirely.** MLM is doing 99.85% of the work. Removing
  saves ~1-2% wall-time and simplifies the model. Lowest risk.
- **Replace with path-bigrams.** Predict the bag of `parent→child` key
  pairs in the masked subtree instead of just keys. Higher cardinality
  (uses path context, not just key identity), so harder. Reuses the
  existing BCE head, just over a richer target vocab.
- **Replace with parent-key prediction.** Given a subtree's vec, predict
  the parent KEY (cross-entropy over atomic vocab). Forces the subtree
  vec to encode "what kind of thing am I a child of." Non-trivial.

Cost: each is a 1-2 day implementation + retraining cycle on v9
infrastructure. The "drop recon" path is cheapest and likely correct
unless someone wants to defend the IDEA of subtree reconstruction.

### 2. Values into doc_vec (first-class aggregation)

v9 made values first-class as attention inputs but kept them out of
direct aggregation. The "values are user payload, doc_vec should be
structure-aware" framing was partially relaxed in v9 but not fully
inverted. The argument in [key-value-design-rationale.md](key-value-design-rationale.md)
acknowledges the empirical channel that values reach doc_vec via
attention, but the aggregator itself is still KEY-only.

A clean v10 experiment: modify the aggregator to mean-pool subwords of
each logical node (already done in v9), then run two paths:
- `doc_vec_structural`: pool KEY logical nodes only (current v9 behavior)
- `doc_vec_full`: pool BOTH KEY and VALUE logical nodes

Train with both heads (one for structural-MLM, one for full-content
recon or similarity). The dual-output gives us back the structural-only
embedding for use cases that need it, while making the full embedding
the default.

Honest blocker: we'd need a use case / failing test that benefits from
content-aware embedding. Retrieval tasks would be the obvious one
(find Deployments running nginx vs running redis), but we don't
currently have a retrieval benchmark to anchor the experiment.

### 3. Absorb values into key subtree_vecs (alternative to #2)

User-proposed alternative: instead of having two doc_vec outputs, let
the aggregator fold a key's child VALUE subwords into its subtree_vec
during combine. So `spec.template.spec.containers[0].resources.requests`
would have a subtree_vec that reflects both its structure AND the
literal memory/cpu values it holds.

Strong version: don't attend across positions to VALUEs at all (restrict
cross-position attention), only let them influence via the local-subtree
aggregator pull-in. More radical, more principled, but a bigger change.

Same blocker as #2 — needs a failing test to evaluate.

### 4. Atomic vocab shrink (quick win)

v9's atomic_target_vocab grew to 11,080 entries (vs v8's 6,049),
inflating the Token Head by ~4M params. Bumping `--min-freq` from 5 to
10 or 15 would halve the atomic vocab and trim the head significantly.

Cost: ~$6 + 12 hours JL training. Acceptance gate: capability test
pass rate within ±5% of v9. Defensible quick win if we want a smaller
model for deployment.

### 5. Tree-bias revival on v9 infrastructure

`yaml_bert/tree_bias.py` was implemented and wired in v8 but disabled
for perf (per-position `attn_mask` forces PyTorch off the fast fused-
attention kernel, training 3-5× slower). Never got a quality verdict.

v9 is a cleaner testing ground because BPE largely fixed the wrong-
parent-pollution problem that tree_bias was originally designed to
address. The question now is whether there's any *residual* structural-
understanding gap that tree_bias closes.

Blocker per [feedback memory "need-failing-test"](../../../.claude/projects/-home-vimal-src-AI-ML-yaml-bert/memory/feedback_need_failing_test.md):
all current tests pass. Before reviving tree_bias, identify a probe or
capability test that fails AND that tree_bias would plausibly fix.
Candidates: deeper-than-15 CRDs, cross-position foreign-key matching
(selector ↔ matchLabels), or something probe-discoverable about
attention pattern quality at depth ≥ 4.

### 6. Real galaxy / probe tool (replace Gradio for viz)

Per [project memory "real-galaxy-tool"](../../../.claude/projects/-home-vimal-src-AI-ML-yaml-bert/memory/project_real_galaxy_tool.md):
the galaxy and structural probes inside the Gradio Space are fighting
the platform. Gradio is right for the missing-field suggester (form in,
suggestion out) but wrong for interactive viz with progressive
disclosure, click-to-side-panel, linked views.

Plan: when v10 lands a doc_vec change (#2 or #3), graduate the viz to
a real interactive tool — D3 / Plotly.js / Observable, hosted separately
from the Space. The Space stays focused on the suggester.

Until then, the Gradio galaxy is "good enough."

### 7. Value subword prediction in MLM

Currently MLM only masks and predicts KEYs. With BPE in place, masking
and predicting VALUE subwords becomes possible. Could be a meaningful
capability boost (model learns value semantics like image-version
patterns, port number distributions, kind values) OR a complexity trap
(doubles the prediction problem, changes the loss balance).

Not a v10 priority unless we have a failing test that benefits from
value-content awareness.

### 8. Article: tokenization → attention

See [project memory "article-tokenization-attention"](../../../.claude/projects/-home-vimal-src-AI-ML-yaml-bert/memory/project_article_tokenization_attention.md).
Side-quest essay on how tokenization shapes what attention can compose,
using the `web-1 / web-2 / web-3` v8→v9 transition as a worked example.
Not a coding task; pure write-up.

---

## Note on scaling

YAML is not language. It's a structured serialization format with
bounded vocabulary, schema-constrained validity, and most manifests
being minor template variations. The data ceiling for YAML-specific
models is much lower than for text — there isn't 10T tokens of
meaningful YAML to train on, and tasks don't require it.

Right scale band for YAML-specialized models: 10M–500M params.
Bigger than current (7.8M), much smaller than frontier LLMs. Don't
try to grow into a YAML-LLM; grow into a better specialized tool that
LLM agents plug into.
