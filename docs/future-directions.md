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

Storage doubles. Status embedding quality unknown empirically (status
has timestamps, counts, free-text messages — none well-handled by a
token model). Worth measuring before committing.

Optional three-way split:
- `intent_vec` (spec + user-written labels/annotations)
- `state_vec` (status only)
- `identity_vec` (name/namespace/ownerRefs — for metadata filtering,
  not similarity)

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
