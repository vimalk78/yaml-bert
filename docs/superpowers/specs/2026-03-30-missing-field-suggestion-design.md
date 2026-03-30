# Missing Field Suggestion

## Problem

Kubernetes YAMLs often miss fields that are not required by the schema but are considered best practice by experienced practitioners. Examples:
- Container without `readinessProbe` (99% of production containers have one)
- Container without `resources.limits` (resource management best practice)
- Deployment without `namespace` (usually means accidental deploy to default)
- Pod without `securityContext` (security best practice)

Schema validators don't catch these — the YAML is valid. Static rule-based linters (like kube-linter) require manually written rules for each check.

## Solution

Use the pre-trained YAML-BERT model as a **convention-based linter**. The model learned from 276K real manifests what keys practitioners typically include at each tree position. Missing high-confidence keys are convention violations.

**No fine-tuning required.** The pre-trained model already knows conventions.

## How It Works

For a given YAML document:

1. **Linearize** the document into nodes
2. **For each key node**, mask it and run the model to get the top-K predicted keys at that position
3. **Collect all predictions** across all positions — this gives us the set of keys the model expects at each tree location
4. **For each tree position**, check if the model's high-confidence predictions actually exist as siblings in the document
5. **Report missing keys** — keys the model expects with high confidence (>50%) that are absent from the document

### Example

Input YAML:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: web
        image: nginx
```

The model predicts at the container level:
- `name` (99%) ✓ present
- `image` (99%) ✓ present
- `resources` (95%) ✗ MISSING
- `ports` (60%) ✗ MISSING
- `readinessProbe` (40%) — below threshold, not reported

Report:
```
MISSING FIELDS:
  spec.template.spec.containers[0]:
    - resources (95% of similar containers have this)
    - ports (60% of similar containers have this)
```

## Algorithm

```python
def suggest_missing_fields(model, vocab, yaml_text, confidence_threshold=0.3):
    nodes = linearize(yaml_text)

    suggestions = []

    # Group nodes by parent_path to find siblings
    siblings_by_parent = group_by(nodes, key=lambda n: n.parent_path)

    for parent_path, sibling_nodes in siblings_by_parent.items():
        # Get existing key names at this level
        existing_keys = {n.token for n in sibling_nodes if n.node_type in (KEY, LIST_KEY)}

        # For each existing key, mask it and get predictions
        # This tells us what keys the model expects at this level
        all_predicted_keys = {}

        for node in sibling_nodes:
            if node.node_type not in (KEY, LIST_KEY):
                continue

            # Mask this key, get top-K predictions
            predictions = mask_and_predict(model, nodes, position=node.index, k=20)

            for key_name, probability in predictions:
                if key_name not in all_predicted_keys:
                    all_predicted_keys[key_name] = probability
                else:
                    all_predicted_keys[key_name] = max(all_predicted_keys[key_name], probability)

        # Find missing keys
        for key_name, probability in all_predicted_keys.items():
            if key_name not in existing_keys and probability >= confidence_threshold:
                suggestions.append({
                    "parent_path": parent_path,
                    "missing_key": key_name,
                    "confidence": probability,
                })

    return sorted(suggestions, key=lambda s: -s["confidence"])
```

## Output Format

```
YAML-BERT Missing Field Report
================================

spec.template.spec.containers[0]:
  [95%] resources — resource limits/requests recommended
  [60%] ports — port definitions commonly present

metadata:
  [45%] namespace — explicit namespace recommended
  [35%] annotations — annotations commonly present

spec.template.spec.containers[0] (when livenessProbe exists):
  [80%] readinessProbe — usually paired with livenessProbe
```

## Confidence Threshold

- **>80%**: Strong recommendation — almost all similar documents have this field
- **50-80%**: Moderate recommendation — majority of similar documents have this
- **30-50%**: Weak recommendation — many similar documents have this
- **<30%**: Not reported

Default threshold: 30%. Configurable via CLI flag.

## Files

| File | Purpose |
|------|---------|
| `yaml_bert/suggest.py` | Core suggestion logic |
| `scripts/suggest_fields.py` | CLI tool |
| `tests/test_suggest.py` | Tests |

## CLI Usage

```bash
# Scan a single file
python scripts/suggest_fields.py checkpoint.pt --yaml-file my-deployment.yaml

# Scan a directory
python scripts/suggest_fields.py checkpoint.pt --yaml-dir ./manifests/

# Adjust threshold
python scripts/suggest_fields.py checkpoint.pt --yaml-file my-pod.yaml --threshold 0.5

# JSON output for tooling integration
python scripts/suggest_fields.py checkpoint.pt --yaml-file my-pod.yaml --format json
```

## What This Does NOT Do (covered by other downstream tasks)

- **Schema validation** — use kubectl for that
- **Value checking** (replicas=9999 is reasonable?) — future downstream task
- **Cross-resource validation** (selectors match?) — Downstream Task 6
- **Security posture scoring** (running as root?) — Downstream Task 4
- **Per-node validity** (is this key valid for this kind?) — Downstream Task 3, requires fine-tuning
- **Best practice compliance report** — Downstream Task 7, builds on this task's output

## Dependencies

- Pre-trained YAML-BERT model (v1 or later)
- Vocabulary file
- No fine-tuning needed
