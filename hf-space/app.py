"""YAML-BERT missing-field suggester — Gradio demo.

Paste a Kubernetes YAML manifest; the model identifies fields it expects
to see but that are absent. Runs the YAML-BERT checkpoint
trained on full 276K K8s corpus — atomic-vocab prediction conditioned on
doc_vec, with tree-aware bottom-up aggregation.

Run locally:
    pip install gradio
    PYTHONPATH=. python app.py
"""
from __future__ import annotations

import os
import sys
import time

# Progress logging — every step on its own line with elapsed wall time, so
# the HF Space build log shows continuous progress instead of long silent gaps.
_T0 = time.time()


def _log(msg: str) -> None:
    print(f"[{time.time() - _T0:6.2f}s] {msg}", file=sys.stderr, flush=True)


_log("Starting app...")
_log("Importing torch (takes a few seconds)...")
import torch
_log(f"torch {torch.__version__} imported")

_log("Importing gradio...")
import gradio as gr
_log(f"gradio {gr.__version__} imported")

_log("Importing yaml_bert package...")
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.suggest import suggest_missing_fields
from yaml_bert.vocab import Vocabulary
_log("yaml_bert imported")


# ----- Model loading (once at startup) -----

DEFAULT_CHECKPOINT = "model/yaml_bert.pt"
DEFAULT_VOCAB = "model/vocab.json"


def load_model(checkpoint_path: str, vocab_path: str) -> tuple[YamlBertModel, Vocabulary]:
    _log(f"Loading vocab from {vocab_path}")
    vocab = Vocabulary.load(vocab_path)
    _log(f"Vocab loaded: {vocab.key_vocab_size} keys, "
         f"{vocab.value_vocab_size} values, "
         f"{vocab.atomic_target_vocab_size} atomic targets")

    _log(f"Reading checkpoint file {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    _log("Building YamlBertModel architecture")
    # recon_enabled=True is required to load checkpoints from MLM+recon training
    # (state_dict includes recon_head weights). The recon head exists but is
    # never invoked at inference time (no subtree_roots_flat passed in forward).
    config = YamlBertConfig(recon_enabled=True)
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModel(
        config=config,
        embedding=emb,
        atomic_vocab_size=vocab.atomic_target_vocab_size,
    )
    _log("Model architecture ready")

    _log("Loading state dict into model")
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    _log("State dict loaded; model in eval mode")
    return model, vocab


checkpoint_path = os.environ.get("YAML_BERT_CHECKPOINT", DEFAULT_CHECKPOINT)
vocab_path = os.environ.get("YAML_BERT_VOCAB", DEFAULT_VOCAB)

MODEL, VOCAB = load_model(checkpoint_path, vocab_path)
n_params = sum(p.numel() for p in MODEL.parameters())
_log(f"Model fully loaded — {n_params:,} parameters")


# ----- Inference -----

import re
import yaml

MAX_LINES_PER_DOC = 300
_DOC_SEP_RE = re.compile(r"^---\s*$", re.MULTILINE)


def _label_for_example(yaml_text: str) -> str:
    """Compact label for an example YAML.

    - Single-doc: 'Kind: name' (with '(namespace)' suffix if metadata.namespace is set)
    - Multi-doc:  'MultiDoc'
    """
    try:
        docs = [d for d in yaml.safe_load_all(yaml_text) if isinstance(d, dict)]
    except Exception:
        return "(unparseable)"
    if len(docs) > 1:
        return "MultiDoc"
    if not docs:
        return "(unidentified)"
    d = docs[0]
    kind = d.get("kind", "(no kind)")
    meta = d.get("metadata") if isinstance(d.get("metadata"), dict) else {}
    name = meta.get("name")
    namespace = meta.get("namespace")
    label = f"{kind}: {name}" if name else str(kind)
    if namespace:
        label += f" ({namespace})"
    return label


def _split_yaml_documents(text: str) -> list[str]:
    """Split a multi-document YAML on '---' separator lines.

    Preserves original formatting (no parse-and-reserialize). Filters out
    empty / whitespace-only chunks and comment-only chunks.
    """
    parts = _DOC_SEP_RE.split(text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        # Skip chunks that are nothing but comments
        if all(line.strip().startswith("#") or not line.strip()
               for line in p.splitlines()):
            continue
        out.append(p)
    return out


def _format_suggestions(suggestions: list[dict]) -> str:
    """Render the suggestions list as a grouped markdown table."""
    by_parent: dict[str, list[dict]] = {}
    for s in suggestions:
        by_parent.setdefault(s.get("parent_path") or "(root)", []).append(s)

    blocks: list[str] = []
    for parent in sorted(by_parent.keys(),
                         key=lambda p: -max(s["confidence"] for s in by_parent[p])):
        rows = ["| Missing key | Confidence | Strength |", "|---|---:|---|"]
        for s in sorted(by_parent[parent], key=lambda x: -x["confidence"]):
            conf = s["confidence"]
            strength = "**STRONG**" if conf >= 0.7 else ("MODERATE" if conf >= 0.5 else "weak")
            rows.append(f"| `{s['missing_key']}` | {conf:.1%} | {strength} |")
        blocks.append(f"#### `{parent}`\n" + "\n".join(rows))
    return "\n\n".join(blocks)


def _detect_kind(yaml_text: str) -> str:
    """Best-effort 'kind:' extraction from raw YAML text (for the header label)."""
    m = re.search(r"^kind:\s*(\S+)", yaml_text, re.MULTILINE)
    return m.group(1) if m else "?"


def _suggest_one(yaml_text: str, threshold: float) -> str:
    n_lines = len(yaml_text.splitlines())
    if n_lines > MAX_LINES_PER_DOC:
        return (
            f"⚠️ **Document too large** — {n_lines} lines (limit: {MAX_LINES_PER_DOC}).\n\n"
            f"The model was trained on manifests up to ~512 linearized nodes. Large "
            f"manifests (cluster-dumped Pods with rich annotations / init containers, deep CRDs) "
            f"exceed that and inference becomes slow and unreliable.\n\n"
            f"Trim verbose sections (annotations, env vars, deep probes) or split the manifest."
        )

    try:
        suggestions, _skipped = suggest_missing_fields(
            MODEL, VOCAB, yaml_text,
            threshold=threshold,
        )
    except Exception as e:
        return f"**Parse error:**\n```\n{e}\n```"

    if not suggestions:
        return ("_No suggestions above threshold — the model thinks this is "
                "either complete or has no strong opinions._")

    return _format_suggestions(suggestions)


def suggest(yaml_text: str, threshold: float) -> str:
    yaml_text = (yaml_text or "").strip()
    if not yaml_text:
        return "_Paste a YAML manifest above to see missing-field suggestions._"

    docs = _split_yaml_documents(yaml_text)
    if not docs:
        return "_No YAML documents found in the input._"

    # Single doc: skip the document header for cleaner output
    if len(docs) == 1:
        return _suggest_one(docs[0], threshold)

    # Multi-doc: prefix each doc's output with a kind/index header
    blocks: list[str] = []
    for i, doc_text in enumerate(docs, 1):
        kind = _detect_kind(doc_text)
        header = f"### Document {i}: `{kind}`"
        blocks.append(header + "\n\n" + _suggest_one(doc_text, threshold))
    return "\n\n---\n\n".join(blocks)


# ----- Manifest galaxy (precomputed UMAP of YAML-BERT doc_vecs) -----

GALAXY_DATA_PATH = "galaxy_data.json"

# Curated qualitative palette + a muted gray for everything outside the top kinds.
_GALAXY_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
]
_GALAXY_OTHER_COLOR = "#d0d0d0"
_GALAXY_TOP_N = 15


def _build_galaxy_figure(data_path: str):
    """Build a Plotly scatter from the precomputed galaxy_data.json."""
    import json
    from collections import Counter
    import plotly.graph_objects as go

    with open(data_path) as f:
        data = json.load(f)

    kinds = data["kind"]
    top_kinds = [k for k, _ in Counter(kinds).most_common(_GALAXY_TOP_N)]
    top_set = set(top_kinds)
    color_map = {k: _GALAXY_PALETTE[i % len(_GALAXY_PALETTE)]
                 for i, k in enumerate(top_kinds)}

    fig = go.Figure()
    # "Other" first so the named kinds render on top
    series: list[tuple[str, str, list[int]]] = []
    other_idxs = [i for i, k in enumerate(kinds) if k not in top_set]
    if other_idxs:
        series.append(("Other", _GALAXY_OTHER_COLOR, other_idxs))
    for k in top_kinds:
        series.append((k, color_map[k], [i for i, kk in enumerate(kinds) if kk == k]))

    for label, color, idxs in series:
        if not idxs:
            continue
        fig.add_scattergl(
            x=[data["x"][i] for i in idxs],
            y=[data["y"][i] for i in idxs],
            mode="markers",
            name=f"{label} ({len(idxs):,})",
            marker=dict(size=4, color=color, opacity=0.65),
            text=[(f"<b>{kinds[i]}</b><br>"
                   f"{data['name'][i]}<br>"
                   f"ns: {data['namespace'][i]}") for i in idxs],
            hovertemplate="%{text}<extra></extra>",
        )

    fig.update_layout(
        title=(f"UMAP projection of {data['n']:,} K8s manifests "
               f"(cosine metric)"),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        height=720,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(orientation="v", x=1.0, y=1.0, font=dict(size=10)),
        plot_bgcolor="white",
    )
    return fig


_log("Building manifest galaxy figure...")
try:
    GALAXY_FIG = _build_galaxy_figure(GALAXY_DATA_PATH)
    _log("Galaxy figure built")
except FileNotFoundError:
    GALAXY_FIG = None
    _log(f"Galaxy data not found at {GALAXY_DATA_PATH} — galaxy tab will be empty")


# ----- UI -----

EXAMPLE_NGINX = """apiVersion: apps/v1
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
"""

EXAMPLE_INCOMPLETE_SERVICE = """apiVersion: v1
kind: Service
metadata:
  name: my-svc
spec:
  selector:
    app: web
"""

EXAMPLE_CONFIGMAP = """apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  key1: value1
"""

EXAMPLE_STATEFULSET = """apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: web
spec:
  serviceName: nginx
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
          name: web
        volumeMounts:
        - name: www
          mountPath: /usr/share/nginx/html
  volumeClaimTemplates:
  - metadata:
      name: www
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 1Gi
"""

EXAMPLE_CRONJOB = """apiVersion: batch/v1
kind: CronJob
metadata:
  name: hello
spec:
  schedule: "*/5 * * * *"
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: hello
            image: busybox:1.28
            command: ["/bin/sh", "-c", "date; echo hello"]
          restartPolicy: OnFailure
"""

EXAMPLE_HPA = """apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: php-apache
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: php-apache
  minReplicas: 1
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 50
"""

EXAMPLE_NETWORKPOLICY = """apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: db-policy
spec:
  podSelector:
    matchLabels:
      app: db
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          role: api
    ports:
    - protocol: TCP
      port: 5432
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
"""

EXAMPLE_INGRESS = """apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: site
spec:
  rules:
  - host: example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: site-svc
            port:
              number: 80
"""

EXAMPLE_POD_INIT_PROBES = """apiVersion: v1
kind: Pod
metadata:
  name: app
spec:
  initContainers:
  - name: init-db
    image: busybox
    command: ["sh", "-c", "until nc -z db 5432; do sleep 1; done"]
  containers:
  - name: app
    image: myapp:1.0
    ports:
    - containerPort: 8080
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 10
      periodSeconds: 5
    readinessProbe:
      httpGet:
        path: /ready
        port: 8080
"""

EXAMPLE_SECRET = """apiVersion: v1
kind: Secret
metadata:
  name: db-credentials
type: Opaque
stringData:
  username: admin
  password: changeme
"""

EXAMPLE_DEPLOYMENT_INCOMPLETE = """# A Deployment missing selector and replicas — model should suggest both
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api
spec:
  template:
    metadata:
      labels:
        app: api
    spec:
      containers:
      - name: api
        image: api:1.0
"""

EXAMPLE_RBAC_MULTIDOC = """# ClusterRole / PSP / ClusterRoleBinding bundle
# Source: kubernetes/examples staging/podsecuritypolicy/rbac/bindings.yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: restricted-psp-user
rules:
- apiGroups:
  - policy
  resources:
  - podsecuritypolicies
  resourceNames:
  - restricted
  verbs:
  - use
---
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: restricted
spec:
  privileged: false
  fsGroup:
    rule: RunAsAny
  runAsUser:
    rule: MustRunAsNonRoot
  seLinux:
    rule: RunAsAny
  supplementalGroups:
    rule: RunAsAny
  volumes:
  - 'emptyDir'
  - 'secret'
  - 'downwardAPI'
  - 'configMap'
  - 'persistentVolumeClaim'
  - 'projected'
  hostPID: false
  hostIPC: false
  hostNetwork: false
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: restricted-psp-users
subjects:
- kind: Group
  apiGroup: rbac.authorization.k8s.io
  name: restricted-psp-users
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: restricted-psp-user
"""

_log("Building Gradio UI...")
with gr.Blocks(title="YAML-BERT") as demo:
    gr.Markdown(
        f"""
# YAML-BERT — structural understanding of Kubernetes YAML

Code: [github.com/vimalk78/yaml-bert](https://github.com/vimalk78/yaml-bert) ·
Trained with MLM + reconstruction on 276K K8s manifests ·
{n_params:,} params

**This Space includes 2 demos — pick a tab below, or use the tiles on the Overview tab.**
"""
    )

    with gr.Tabs() as tabs:
        with gr.Tab("Overview", id="overview"):
            gr.Markdown("### Demos in this Space")
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        "#### 🧩 Missing-field suggester\n"
                        "Paste a Kubernetes YAML manifest. The model walks each "
                        "parent level, identifies fields it expects to see there "
                        "but that are absent, and ranks the suggestions by "
                        "confidence."
                    )
                    open_suggester = gr.Button(
                        "Open missing-field suggester →", variant="primary"
                    )
                with gr.Column():
                    gr.Markdown(
                        "#### 🌌 Manifest galaxy\n"
                        "10,000 Kubernetes manifests embedded by the model and "
                        "projected to 2D. Watch clusters of `Deployment`, "
                        "`Service`, `ConfigMap` etc. form spontaneously — the "
                        "model was never told what `kind` is."
                    )
                    open_galaxy = gr.Button(
                        "Open manifest galaxy →", variant="primary"
                    )

        with gr.Tab("Missing-field suggester", id="suggester"):
            gr.Markdown(
                "Paste a Kubernetes YAML manifest. The model walks each "
                "parent level, identifies fields it expects to see there "
                "but that are absent, and ranks the suggestions by confidence."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    yaml_input = gr.Code(
                        language="yaml",
                        lines=22,
                        max_lines=100,
                        label="YAML input",
                        value=EXAMPLE_NGINX,
                    )
                    threshold = gr.Slider(
                        minimum=0.05, maximum=0.95, value=0.1, step=0.05,
                        label="Confidence threshold",
                    )
                    submit = gr.Button("Suggest missing fields", variant="primary")

                with gr.Column(scale=1):
                    output = gr.Markdown(label="Suggestions", value="")

            submit.click(fn=suggest, inputs=[yaml_input, threshold], outputs=output)
            # No auto-trigger on yaml_input.change — typing/pasting a long YAML
            # would fire many inference requests and back up the queue.

            _ALL_EXAMPLES = [
                EXAMPLE_NGINX,
                EXAMPLE_DEPLOYMENT_INCOMPLETE,
                EXAMPLE_INCOMPLETE_SERVICE,
                EXAMPLE_CONFIGMAP,
                EXAMPLE_SECRET,
                EXAMPLE_STATEFULSET,
                EXAMPLE_CRONJOB,
                EXAMPLE_HPA,
                EXAMPLE_NETWORKPOLICY,
                EXAMPLE_INGRESS,
                EXAMPLE_POD_INIT_PROBES,
                EXAMPLE_RBAC_MULTIDOC,
            ]
            gr.Examples(
                examples=[[y] for y in _ALL_EXAMPLES],
                example_labels=[_label_for_example(y) for y in _ALL_EXAMPLES],
                inputs=[yaml_input],
                examples_per_page=20,
                label="Example YAMLs",
            )

            gr.Markdown(
                """
---

### What it does well
- Predicts standard Kubernetes structural fields
- Distinguishes kind-specific fields (`Deployment.replicas` vs `Service.ports`)
- Calibrated confidence: strong on common patterns, weaker in ambiguous positions

### Known limitations
- Status-side fields are not well predicted
- Novel CRD instances and rare annotation keys may not work
- Trained on `substratusai/the-stack-yaml-k8s`
"""
            )

        with gr.Tab("Manifest galaxy", id="galaxy"):
            gr.Markdown(
                "Each point is one K8s manifest from the training corpus, "
                "embedded as a `doc_vec` by the bottom-up tree aggregator, "
                "then projected to 2D with UMAP (cosine metric). "
                "Manifests with similar structure end up near each other — "
                "the model has never been told what `kind` is, "
                "yet clusters of `Deployment`, `Service`, `ConfigMap` etc. "
                "form spontaneously."
            )
            if GALAXY_FIG is not None:
                gr.Plot(value=GALAXY_FIG, show_label=False)
            else:
                gr.Markdown("_Galaxy data unavailable._")
            gr.Markdown(
                "Hover for `kind / name / namespace`. "
                "Top 15 kinds are colored; everything else is gray. "
                "Click a legend entry to toggle that kind on/off."
            )

    open_suggester.click(lambda: gr.Tabs(selected="suggester"), outputs=tabs)
    open_galaxy.click(lambda: gr.Tabs(selected="galaxy"), outputs=tabs)


_log("Gradio UI built — launching")

if __name__ == "__main__":
    demo.launch()
