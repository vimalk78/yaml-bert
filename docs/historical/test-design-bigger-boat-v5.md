# Bigger Boat — Test Design

## Premise

v5 passes 93/93 pretrain capability tests, 6/9 structural tests, and runs
the `suggest_fields` tool with useful results on real manifests. Across
all ablation variants (full, no_depth, no_sibling, sequential) the
capability suite saturates at 91–93/93 — it can no longer differentiate
trained models.

The question this document answers: **what's actually missing from our
test coverage that v6 should be designed to address?**

The answer turns out to be much narrower than a naïve read suggests. The
current test surface is dense; the gaps are specific.

## What we already have

### `model_tests/test_capabilities.py` — 30 capabilities, 121+ tests

Covers structural recall, kind conditioning, required fields (universal
+ kind-specific via *Kind-specific spec children*), wrong-context
rejection (universal + kind-specific via *Kind-specific invalid structure
rejection*), workload patterns (Multi-container awareness, Workload
controller distinction, plus dedicated capabilities for StatefulSet,
DaemonSet, Job, HPA), volume/probe/service-port enum completion,
security context level-awareness, scheduling/affinity, RBAC structure,
PV/PVC, Ingress, ConfigMap vs Secret, value-context sensitivity,
container field completeness, annotation patterns.

Two capabilities are flagged `phase="finetune"` — Invalid structure
rejection and Kind-specific invalid structure rejection — both
calibration-flavored (test that the model's confidence drops at wrong
positions).

### `model_tests/test_structural.py` — 9 hand-built tests

Kind conditioning, wrong parent rejection, depth awareness, spec-vs-
status, nonsense-YAML calibration, missing required field. v5 passes
6/9; the 3 known failures are:
- spec/status distinction (model predicts [UNK] at 99%)
- one missing-field test
- (third failure I should verify)

### `model_tests/test_bigger_boat.py` — 13 tests across 4 categories

Status-side completion (4 — all fail), CRD pollution (4 — all pass),
annotation keys (2 — pass), confidence calibration (3 — pass). v5 scores
9/13 = 69%.

### `scripts/suggest_fields.py` — downstream tool

Not a test but a working use case. Probes each parent level by inserting
a synthetic `[MASK]` and reading predictions. Categorizes by confidence
(STRONG/MODERATE/WEAK). Detects *wrong-level predictions* where the
predicted compound target's parent doesn't match the probed position —
this surfaces a real model weakness already.

## What's actually missing

Six narrow gaps, in rough order of importance:

### Gap 1 — Systematic status-side completion

**What we have:** 1 test in test_structural.py (fails); 4 tests in
test_bigger_boat.py vocab_gap category (all fail). Confirmed [UNK]
collapse for `Deployment.status.replicas`, `Pod.status.conditions`,
`HPA.status.desiredReplicas`, `Service.status.loadBalancer.hostname`.

**What's missing:** wider coverage across kinds and across the status
subtree shape (conditions[i].type, containerStatuses[i].imageID,
loadBalancer.ingress[i].ip, podIP/hostIP, capacity, succeeded/failed
counters, etc.).

**Why it matters:** the gap is real and measurable; addresses Lever 1
in v6 plan; also the dominant downstream blind spot when developers want
to write status-aware tooling on top of the model.

**Tests to add:** 6–8 across `StatefulSet.status`, `Job.status`,
`Pod.status.containerStatuses`, `Node.status.conditions`,
`Service.status.loadBalancer.ingress`, `PVC.status.capacity`.

### Gap 2 — API-version awareness

**What we have:** nothing. No test capability mentions `apiVersion`
variants. The `kind_head` is conditioned on `kind` alone, not on
`apiVersion`.

**Why it matters:** real corpora include deprecated API versions
(`extensions/v1beta1` Ingress, `apps/v1beta1` Deployment, `autoscaling/v1`
HPA with `targetCPUUtilizationPercentage` vs `autoscaling/v2` with the
`metrics` array). The model probably mixes them.

**Tests to add:** 3–4 paired tests — same kind under v1 vs vbeta1,
verifying version-specific fields are predicted.

**v6 plan implication:** add **Lever 6 — apiVersion-aware kind head**
(see v6-plan.md update).

### Gap 3 — Wrong-level / wrong-sibling predictions, scored

**What we have:** `suggest_fields.py` detects this during probing
("predicted parent doesn't match probed parent"), reports it in the demo
output. But there is no test capability that scores wrong-level
prediction rate as a metric.

**Why it matters:** this is documented in `next-training-improvements.md`
as motivating *Lever 6 in that doc — tree-aware attention bias*. Worth
turning the detection into a scored test.

**Tests to add:** 4–5 cases that mask at one position but provide a
neighboring well-formed subtree; expect the predicted compound target's
parent to match the probed parent.

### Gap 4 — CRD-instance handling

**What we have:** the corpus is heavy on CRD definitions (3% of docs,
46% of training tokens). We have zero tests on CRD *instances* — the
custom resources whose shape the CRDs define.

**Why it matters:** users actually deploy these resources (Prometheus
objects, cert-manager Issuers, ArgoCD Applications, Tekton Pipelines,
etc.). If v6 reweighting changes how CRDs are seen during training, we
need a way to check whether downstream behavior on CRD instances stays
useful.

**Tests to add:** 4–6 manifests of common custom resources (with no
schema content — just the instance), masked at typical positions.

**Caveat:** v5 has likely never seen these as instances; this is
*genuine* OOV testing, not "we trained on it and want to confirm."
Expectations should be lower (~50% pass rate is meaningful).

### Gap 5 — Adversarial OOD calibration

**What we have:** test_structural.py #5 — nonsense YAML drops confidence
from 100% to 66% — exactly one test.

**What's missing:**
- **Typo correction** — `containres` instead of `containers`. Currently
  the model has no chance because tokens are atomic; this test would
  document a v7-sub-tokenization gap.
- **Non-K8s YAML** — feed a Helm values file, a CI config, a docker-
  compose. Model should produce LOW confidence; should not confidently
  emit K8s structural keys.
- **Truncated YAML** — only first few lines (apiVersion + kind + half of
  metadata). Predictions should still be reasonable.

**Tests to add:** 3–4 OOD/adversarial cases.

### Gap 6 — Confidence floors on existing tests

**What we have:** most capability tests use `expect_in_top5` without any
confidence requirement. So a required field at rank 5 with 3% confidence
passes the same as at rank 1 with 99%.

**Why it matters:** for the `suggest_fields` downstream use, top-5
inclusion is too permissive. We want to know *how confident* the model is
about required fields specifically.

**What's missing:** tightened assertions on a subset of existing tests.
Not new YAMLs — just stricter expectations on `expect_confidence_above`
for tests of well-known required fields.

**Cost:** ~20 lines of edits across existing capabilities, no new tests.

## What's NOT missing (avoid duplicating)

For honesty, here's what I previously thought was missing but actually
exists:

- **Universal required fields** — Capability 5, with high-confidence
  thresholds already
- **Kind-specific required fields** — capabilities 12–19 + 21
  (Kind-specific spec children) cover StatefulSet.serviceName,
  DaemonSet.updateStrategy, HPA.scaleTargetRef, PodDisruptionBudget,
  StorageClass, ResourceQuota
- **Mutual exclusivity** — capabilities 6 + 22 (Invalid structure
  rejection variants) + Volume semantics + Probe structure handle this
- **Volume / probe / service-port enum completion** — capabilities 11,
  15, 17
- **Workload patterns** — capabilities 9, 12, 13, 14, 25 + ConfigMap vs
  Secret + Ingress + PV/PVC
- **RBAC structure** — capability 10
- **Long-tail kinds** — capabilities 19 (HPA), 25+ via Kind-specific
  spec children additions (NetworkPolicy, PodDisruptionBudget,
  StorageClass, ResourceQuota, etc.)
- **Annotation keys (basic)** — capabilities 20 + 30

## Sizing the buildout

| Gap | New tests | Effort |
|---|---|---|
| 1. Status completion | 6–8 | ~1 day (handcrafting YAMLs) |
| 2. API version | 3–4 | ~half day |
| 3. Wrong-level (scored) | 4–5 | ~half day |
| 4. CRD-instance | 4–6 | ~1 day (need real CRD instance examples) |
| 5. Adversarial OOD | 3–4 | ~half day |
| 6. Confidence floors | 0 new YAMLs | ~2 hours (tightening assertions) |

**Total: ~20–25 new tests over ~3 days of work.** Manageable, focused.

## Build order

Order by what most directly informs v6 evaluation:

1. **Gap 1 (status completion)** first — it's the failure mode v6 Lever 1
   targets. Need a comprehensive measure to know if Lever 1 worked.
2. **Gap 6 (confidence floors)** second — cheap, tightens existing tests,
   raises the bar on what "passing" means before v6.
3. **Gap 2 (API version)** — motivates adding Lever 6 to v6 plan.
4. **Gap 5 (OOD calibration)** — sanity for any v6 (we don't want
   confidence to *worsen* outside the K8s distribution).
5. **Gap 4 (CRD instances)** — checks that v6 reweighting doesn't break
   generalization to actual custom resources.
6. **Gap 3 (wrong-level scored)** — motivates the v7 tree-attention-bias
   work documented in `next-training-improvements.md`.

## Pass thresholds for v6 evaluation

| Gap | v5 baseline | v6 target |
|---|---|---|
| 1. Status completion | 0/8 | ≥ 6/8 |
| 2. API version | unmeasured (v5 unknown) | ≥ 50% |
| 3. Wrong-level (scored) | unmeasured | ≥ 60% |
| 4. CRD-instance | unmeasured | ≥ 50% (OOV, lower bar) |
| 5. OOD calibration | partial (1 test pass) | ≥ 75% |
| 6. Confidence-floor reruns of existing tests | TBD after tightening | no regression |

Plus the unchanged criteria:
- `test_capabilities.py` should remain at 93/93 (no regression)
- `test_structural.py` should improve from 6/9 → 8+/9 (status failure closed)

## Compelling presentation arc this enables

Without this rubric: "we did ablations, found the test suite saturates."

With this rubric: "We characterized v5 against 6 specific blind spots,
designed targeted v6 interventions for each, and measured per-gap deltas
to validate the design."

That's a research-to-engineering arc — much sharper than either alone.
