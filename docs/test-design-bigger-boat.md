# Bigger Boat — Test Design

The current saturated capability suite (`test_capabilities.py`, 93/93) measures
*basic structural recall* — does the model know `replicas` goes under `spec`?
This passes for any reasonably-trained model.

The bigger-boat suite must instead measure whether the model has learned the
**operating semantics** of K8s YAML — the kind of understanding a competent
operator brings to a manifest. This document enumerates the dimensions that
"understands K8s YAML" actually covers, the test categories that probe each,
and the difficulty floor each implies for the architecture.

## What a model that "knows K8s YAML" should be able to do

1. **Structural recall** — basic key-to-parent mapping
   (passing the existing capability suite; v5 ✓ 93/93)
2. **Required-field awareness** — knows which keys are required vs optional
   per kind; can complete missing required fields
3. **Mutual exclusivity / conditional schemas** — knows which fields exclude
   each other (e.g., probe types, env value sources)
4. **API version awareness** — distinguishes equivalent kinds across versions
   (e.g., `apps/v1` vs deprecated `apps/v1beta1`)
5. **Defaulting behavior** — knows what gets defaulted when omitted
   (e.g., `restartPolicy: Always` for Deployments)
6. **Status-side recall** — predicts status keys, not just spec keys
   (v5 ✗ — known gap)
7. **Sibling cross-reference** — within one document, knows that paired
   names must match (`volumes.name` ↔ `volumeMounts.name`)
8. **Workload-pattern fluency** — recognizes common multi-container, sidecar,
   initContainer patterns
9. **Probe / volume / scheduling type completion** — knows the small enum
   of valid types in each slot
10. **RBAC structural consistency** — knows the constrained shape of rules,
    verbs, resources, apiGroups
11. **Long-tail kind handling** — handles kinds rarely seen in training
12. **Annotation / label namespace awareness** — handles user-extensible
    metadata slots reasonably
13. **Calibration** — confident on clear, uncertain on ambiguous
    (v5 ✓ on the few we tested)
14. **Anti-pattern recognition** — flags `:latest` tags, missing health
    checks, privileged containers (aspirational)
15. **Cross-document referential consistency** — Service selector matches
    Pod labels (architecturally out of scope for a single-doc model)

Items 14 and 15 are *aspirational* — out of reach for the current
single-document encoder. The rest are reachable with current architecture
and worth testing.

## Test categories — concrete proposals

For each, I list: what it probes, why it matters, 2–3 example tests, and
difficulty floor (whether v5 likely passes/fails).

### Category A — Required-field completion (NEW)

**Probes:** item 2 (required-field awareness).

**Why it matters:** the primary downstream use case is `suggest_fields` —
"what's missing from my YAML?". A model that's confident here is directly
useful.

**Example tests:**
- Deployment with no `selector` → mask the position; expect top-1 `selector`
  with high confidence
- Pod with no `spec.containers` (only metadata) → expect `spec` or
  `containers` highly ranked
- Service with no `spec.ports` → expect `ports`
- StatefulSet with no `serviceName` → expect `serviceName`
- Job with neither `completions` nor `parallelism` → expect both in top-5
- PersistentVolumeClaim with no `accessModes` → expect `accessModes`

**Difficulty floor:** v5 should pass these on common kinds. Failure means
the model only knows what fields *can* exist, not what *must* exist.

### Category B — Mutually exclusive / conditional schemas (NEW)

**Probes:** item 3.

**Why it matters:** real schemas have "exactly one of" constraints. Models
that don't know this produce confidently wrong suggestions.

**Example tests:**
- Container env from `valueFrom` already specifies `configMapKeyRef`; mask
  the *sibling* position — should NOT suggest `secretKeyRef` or `fieldRef`
- Pod with `nodeName` set; mask another position — should NOT suggest
  `nodeSelector`
- Liveness probe with `exec` set; mask sibling — should NOT suggest
  `httpGet` or `tcpSocket`
- Volume with `hostPath` set; mask sibling — should NOT suggest `emptyDir`,
  `configMap`, etc.

**Difficulty floor:** v5 likely fails some of these. The probe slot is
particularly testable (3-way mutual exclusion).

### Category C — Volume / probe / scheduling type completion (NEW)

**Probes:** item 9 — enum completion in specific slots.

**Example tests:**
- Mask `httpGet` under `readinessProbe`; expect alternatives like `exec`,
  `tcpSocket` in top-5
- Mask `hostPath` under volumes; expect `emptyDir`, `persistentVolumeClaim`,
  `configMap`, `secret`, `projected` in top-5
- Mask `topologyKey` under affinity; expect `kubernetes.io/hostname`,
  `topology.kubernetes.io/zone`

**Difficulty floor:** medium. Tests whether the model has the *closed enum*
internalized.

### Category D — Status-side completion (EXISTS, expand)

**Probes:** item 6.

**Example tests already in v1 bigger boat (all fail on v5):**
- `Deployment.status.replicas`, `Pod.status.conditions`,
  `HPA.status.desiredReplicas`, `Service.status.loadBalancer.hostname`

**Additional tests to add:**
- `StatefulSet.status.currentReplicas` / `readyReplicas`
- `Job.status.succeeded` / `failed` / `active`
- `Pod.status.containerStatuses[].imageID`
- `Node.status.conditions[].type` (DiskPressure, MemoryPressure, etc.)

**Difficulty floor:** v5 fails all. The v6 plan's Lever 1 (selective
masking) should close most.

### Category E — API-version awareness (NEW)

**Probes:** item 4.

**Why it matters:** real manifests live across versions. Model should know
the canonical version-specific fields.

**Example tests:**
- `apiVersion: networking.k8s.io/v1` Ingress: `pathType` is required;
  v1beta1 doesn't have it
- `apiVersion: autoscaling/v2` HPA: `metrics` array; `autoscaling/v1` has
  `targetCPUUtilizationPercentage` instead
- `apiVersion: apps/v1` Deployment: `selector` required; older versions
  defaulted it

**Difficulty floor:** medium-hard. v5 might fail when the rarer version
appears.

### Category F — Sibling cross-reference (NEW)

**Probes:** item 7 — within-doc referential consistency.

**Example tests:**
- Mask `volumeMounts[i].name` after volumes have been declared with name
  "config", "data" — expect those exact strings in top-5
- Mask `secretKeyRef.name` after a Secret has been referenced earlier in
  the same doc — expect that exact name

**Difficulty floor:** hard for current architecture (it's a key-prediction
model, not a copy-from-context model). Likely fails — but quantifying the
gap motivates v7 features.

### Category G — Workload-pattern fluency (NEW)

**Probes:** item 8.

**Example tests:**
- Deployment with `initContainers` + main `containers` + `volumes` — mask
  positions in each section, all should be appropriately predicted
- Pod with sidecar container (multi-`containers`) — verify model handles
  per-container `env`, `volumeMounts`, `resources`
- StatefulSet with `volumeClaimTemplates` (different from regular `volumes`)
- DaemonSet (no `replicas`!) — model should NOT suggest replicas

**Difficulty floor:** v5 likely passes the common ones; the DaemonSet
"don't suggest replicas" test is interesting.

### Category H — RBAC consistency (NEW)

**Probes:** item 10.

**Example tests:**
- Role with `rules` declared; mask `verbs` — expect `get`, `list`, `watch`,
  `create`, `update`, `patch`, `delete` in top-5 (verb enum)
- Role with `apiGroups: [""]`; mask `resources` — expect `pods`, `services`,
  `configmaps`, ... (core API resources)
- RoleBinding `subjects[].kind` — expect `ServiceAccount`, `User`, `Group`

**Difficulty floor:** medium. RBAC was 4.7% of v5's training (ClusterRole
+ Role + bindings ≈ 10%), so model should know the closed enums.

### Category I — Long-tail kinds (NEW)

**Probes:** item 11.

**Example tests:**
- NetworkPolicy with `policyTypes: [Ingress, Egress]` (NetworkPolicy was
  0.6% of training)
- PodDisruptionBudget with `minAvailable` and `selector`
- ResourceQuota with `hard` map
- LimitRange with `limits` array containing type-specific defaults

**Difficulty floor:** unknown. Some long-tail kinds might have surprising
gaps.

### Category J — Annotation / label namespace (EXISTS, expand)

**Probes:** item 12. Current bigger boat has 2 tests; expand to ~5.

**Examples to add:**
- `helm.sh/chart`, `helm.sh/hook`, `helm.sh/hook-weight`
- `argocd.argoproj.io/sync-wave`, `argocd.argoproj.io/sync-options`
- `meta.helm.sh/release-name`
- `kubernetes.io/ingress.class` (deprecated path)
- Custom `mycompany.com/owner: team-x` style annotations (truly novel)

**Difficulty floor:** v5 passes 2/2 today; novel annotations will fail
until sub-tokenization (v7).

### Category K — Calibration extremes (EXISTS, expand)

**Probes:** item 13.

**Tests to add:**
- **Adversarially shaped YAML** — a Pod where `spec.containers` is replaced
  by `spec.containres` (typo); mask the typo'd key. Expect the model to
  suggest `containers` (typo correction) with > 50% confidence.
- **Wildly out-of-distribution YAML** — non-K8s YAML (a Helm chart values
  file, a CI config). Model should produce LOW confidence; not confidently
  emit K8s structural keys.

**Difficulty floor:** typo-correction is likely an honest failure (tokens
are atomic — model won't see partial matches without sub-tokenization).
OOD-confidence might pass.

## Coverage of current bigger boat

The existing `test_bigger_boat.py` covers 4 categories partially:

| Category | Current count | Target for v6 |
|---|---|---|
| D — status-side completion | 4 (all fail) | 8–10 |
| Probably-CRD-pollution | 4 (all pass) | keep or drop |
| J — annotation keys | 2 (pass) | 5–8 |
| K — calibration | 3 (pass) | 6–8 (add typo + OOD) |
| A — required fields | 0 | 6 |
| B — mutual exclusivity | 0 | 5 |
| C — type completion | 0 | 4 |
| E — API version | 0 | 4 |
| F — sibling cross-ref | 0 | 3 (likely fail, useful gap) |
| G — workload patterns | 0 | 5 |
| H — RBAC | 0 | 4 |
| I — long-tail kinds | 0 | 4 |

Target for v6 evaluation: **~60 tests across 11 categories**.

## What v6 must pass — prioritized

For each category, what would success look like and what's the architecture
implication:

### Must pass (or v6 is a regression)

- **A. Required fields** — `≥ 80%`. Direct utility for suggest_fields.
- **D. Status-side completion** — `≥ 75%`. The known gap. Lever 1
  (selective masking) targets this directly.
- **G. Workload patterns** — `≥ 80%`. Common in production.
- **H. RBAC** — `≥ 70%`. Closed enums should be learnable.
- **J. Annotation keys** — `≥ 60%` and importantly *not [UNK]* as top-1.
  Levers 3+4 address this.

### Stretch (motivates v7)

- **B. Mutual exclusivity** — `≥ 50%`. Hard but achievable with kind/
  parent context.
- **E. API version awareness** — `≥ 50%`. Needs apiVersion-conditional
  routing in the kind head.
- **C. Type completion** — `≥ 70%`. Enum knowledge.

### Out of scope for v6 (architecture-bound)

- **F. Sibling cross-reference** — current model can't copy from context.
  Document the gap; defer to v7 (attention-bias variant).
- **K — typo correction** — needs sub-tokenization. Defer to v7.

## How this changes the v6 plan

Looking at the levers in `v6-plan.md` against this test rubric:

| Lever | Category it most helps |
|---|---|
| 1 — Selective masking | D (status), some A (required) |
| 2 — Loss reweighting | indirect — rebalances toward real manifests, helps everything |
| 3 — Per-parent min_freq | J (annotations) |
| 4 — Annotation head | J (annotations) — better than Lever 3 alone |
| 5 — CRD doc cap | indirect, like Lever 2 |

**Gaps in the v6 plan that this rubric exposes:**

- **Category A (required fields)** — none of the v6 levers directly target
  this. v5 likely passes most but worth measuring; if any fail, may need a
  *required-field-emphasized* training objective (e.g., always mask the
  position where a required field should be).
- **Category E (API version)** — apiVersion is currently in input but the
  kind_head is conditioned only on `kind`, not on `apiVersion`. A small
  arch tweak (concatenate apiVersion to the kind context) could help.
- **Category H (RBAC)** — closed enums. Loss reweighting (Lever 2) should
  help indirectly; otherwise no targeted lever needed.

**Suggested v6 extension:** add a sixth lever —

### Lever 6 — apiVersion-aware kind head (NEW idea, derived from rubric)

**What.** Currently `kind_head` outputs targets like
`Deployment::spec::replicas`. Extend to `apps/v1::Deployment::spec::replicas`,
so the head distinguishes versions. Targets currently rare under one version
gain support from the equivalent target in another.

**Why.** Category E demands version awareness. Trivial cost.

**Cost.** ~20 lines in vocab + dataset.

## Recommended build order

1. **First, expand the bigger boat** to ~30 tests across Categories A, B, D,
   G, H, J (the practical, must-pass ones). Run against v5 — get baseline.
2. **Then implement v6 Phase 1** (Levers 1 + 2 + 6).
3. **Re-run expanded bigger boat** on v6.1 — measure improvement
   per-category, identify remaining gaps.
4. **Then v6 Phase 2** (Levers 3 + 4) if annotation handling is the
   remaining gap.

The compelling presentation arc is:
"We tested whether the model knows K8s YAML — not just structurally but
operationally. Found 5 systematic failure modes via embedding analysis and
extended testing. Designed 6 targeted interventions for v6. Here are the
gains we measured."
