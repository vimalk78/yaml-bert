# Model Tests

Behavioral tests that verify what the model learned, not code correctness (that's in `tests/`).

## test_capabilities.py — 121 test cases, 30 capabilities

The main evaluation. Each test masks a specific key in a handcrafted YAML and checks if the model predicts the correct key in its top 5. Capabilities are grouped by what they test:

- **Parent-child validity** — does the model know which keys belong under which parent?
- **Kind conditioning** — does masking a key under `spec` produce different predictions for Deployment vs Service vs ConfigMap?
- **Depth sensitivity** — does the model predict root keys at depth 0 and nested keys at depth 3+?
- **Sibling awareness** — if `limits` exists, does the model predict `requests` as its sibling?
- **Required fields** — are `apiVersion`, `metadata`, `name` predicted with high confidence?
- **Invalid structure rejection** — does the model have low confidence when keys are in wrong positions?
- **Cross-kind discrimination** — does it know Secret has `type`, PVC has `accessModes`, Ingress has `rules`?
- **Value-context sensitivity** — do unmasked values (like port numbers) influence key predictions?
- **Kind-specific structure** — 15 capabilities testing Deployments, StatefulSets, DaemonSets, Jobs, CronJobs, Services, ConfigMaps, Secrets, RBAC, HPA, Ingress, PV/PVC, Probes, SecurityContext, Scheduling

## test_structural.py — 9 tests

Targeted structural reasoning tests:

1. **Kind conditioning** — same key, different kinds produce different predictions
2. **Wrong parent** — `containers` under `metadata` should predict metadata keys, not `containers`
3. **Depth awareness** — depth 0 predicts root keys, depth 4 predicts nested keys
4. **spec vs status** — `replicas` under `spec` vs `status` (currently fails due to vocab gap)
5. **Nonsense YAML** — invalid structure produces lower confidence than valid
6. **Missing required field** — masking where `metadata` should be

## Running

```bash
# Both tests require a checkpoint and vocab
PYTHONPATH=. python model_tests/test_capabilities.py <checkpoint> --vocab <vocab>
PYTHONPATH=. python model_tests/test_structural.py <checkpoint> --vocab <vocab>

# Or run everything at once
./scripts/run_all_tests.sh <checkpoint> <vocab>
```

See [evaluation results](../docs/evaluation-results.md) for latest output.
