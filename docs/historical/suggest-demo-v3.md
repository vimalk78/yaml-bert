# Missing Field Suggestion — Demo Output

The suggest tool probes each level of a YAML document by inserting a fake `[MASK]` node and reading the model's predictions. If the model predicts a key that doesn't already exist in the document, it's a candidate for a missing field.

## How it works

1. Parse the YAML into a tree of nodes
2. For each parent level (root, metadata, spec, spec.template, etc.), insert a `[MASK]` as the next sibling after the last child
3. Run the model forward — it predicts what key should go at that position
4. If the predicted key doesn't exist in the document, report it as a suggestion
5. The compound target (e.g., `spec::strategy`) tells us both the predicted key AND which parent it belongs to — if the predicted parent doesn't match where we placed the mask, the prediction is flagged as a wrong-level prediction

## Example: nginx Deployment

Input YAML (`testdata/deployment/deployment-nginx.yaml`):

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
spec:
  selector:
    matchLabels:
      app: nginx
  replicas: 2
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.8
        resources:
          limits:
            memory: "128Mi"
            cpu: "250m"
        ports:
        - containerPort: 80
```

## Output

```
spec.template.spec.containers.0.resources:
    [99.8%] requests (STRONG)

spec:
    [99.4%] strategy (STRONG)

metadata:
    [96.7%] labels (STRONG)

spec.template.spec.containers.0.ports.0:
    [95.6%] name (STRONG)

spec.template.spec:
    [52.2%] imagePullSecrets (MODERATE)

spec.template.spec.containers.0:
    [45.4%] args (WEAK)

Wrong-level predictions (parent mismatch, model weakness):
    spec.selector: predicted strategy::rollingUpdate (51.4%)
        — expected parent 'selector', model predicted parent 'strategy'
    spec.selector.matchLabels: predicted spec::imagePullSecrets (52.2%)
        — expected parent 'matchLabels', model predicted parent 'spec'
    spec.template.metadata: predicted spec::imagePullSecrets (52.2%)
        — expected parent 'metadata', model predicted parent 'spec'

Total: 6 suggestions
```

## Interpretation

**Good suggestions (model correctly identified missing fields):**

| Suggestion | Why it's correct |
|-----------|-----------------|
| `resources.requests` (99.8%) | The YAML has `limits` but no `requests` — best practice is to set both |
| `spec.strategy` (99.4%) | Deployments should specify an update strategy (RollingUpdate or Recreate) |
| `metadata.labels` (96.7%) | The top-level metadata has `name` but no `labels` — labels are standard |
| `ports.name` (95.6%) | Named ports are required for service mesh compatibility |

**Moderate/weak suggestions:**

| Suggestion | Assessment |
|-----------|-----------|
| `imagePullSecrets` (52.2%) | Valid but not always needed — depends on registry auth |
| `args` (45.4%) | Valid container field but nginx image doesn't need args |

**Wrong-level predictions (model weakness):**

The model sometimes predicts keys for a neighboring subtree instead of the probed position. For example, when probing under `selector.matchLabels`, the model predicts `spec::imagePullSecrets` — a valid key, but for the wrong parent. The compound target makes this detectable: the predicted parent (`spec`) doesn't match the probed parent (`matchLabels`).
