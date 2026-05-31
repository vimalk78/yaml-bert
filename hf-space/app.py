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
    _log(f"Vocab loaded: subword={vocab.subword_vocab_size}, "
         f"atomic targets={vocab.atomic_target_vocab_size}")

    _log(f"Reading checkpoint file {checkpoint_path}")
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    _log("Building YamlBertModel architecture (v9: subword embedding)")
    # recon_enabled=True keeps the checkpoint's recon_head weights loadable.
    # The recon head exists but is never invoked at inference time
    # (no subtree_roots_flat passed in forward).
    config = YamlBertConfig(recon_enabled=True)
    emb = YamlBertEmbedding(
        config=config,
        subword_vocab_size=vocab.subword_vocab_size,
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

# Qualitative palette (Plotly D3-style category20 + 5 more) so we can color
# up to 25 distinct kinds before falling back to "Other". The corpus has
# 49 distinct kinds total; the long tail under ~25 is rare enough that
# graying them together is fine.
_GALAXY_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
]
_GALAXY_OTHER_COLOR = "#d0d0d0"
_GALAXY_TOP_N = 25


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

    def _hover(i: int) -> str:
        parts = [f"<b>{kinds[i]}</b>", data["name"][i]]
        ns = data["namespace"][i]
        # "(default)" is the build-time placeholder when the manifest didn't
        # specify a namespace — true for all cluster-scoped kinds (Namespace,
        # ClusterRole, CRD, …) and for namespaced resources that omit it.
        # Showing "ns: (default)" for these is misleading, so suppress.
        if ns and ns != "(default)":
            parts.append(f"ns: {ns}")
        return "<br>".join(parts)

    for label, color, idxs in series:
        if not idxs:
            continue
        fig.add_scattergl(
            x=[data["x"][i] for i in idxs],
            y=[data["y"][i] for i in idxs],
            mode="markers",
            name=f"{label} ({len(idxs):,})",
            marker=dict(size=4, color=color, opacity=0.65),
            text=[_hover(i) for i in idxs],
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


# ----- Structural probes -----
#
# Presets of hand-crafted manifests that probe whether the model encodes
# specific structural distinctions. Each preset's manifests are encoded to
# doc_vecs, projected to 2D via MDS (preserves pairwise cosine distances),
# and shown as a scatter plot. Users can add their own YAMLs to see where
# they land relative to the preset.

# ---- Preset manifests ----

_POD_NGINX = """apiVersion: v1
kind: Pod
metadata:
  name: nginx-app
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_POD_REDIS = """apiVersion: v1
kind: Pod
metadata:
  name: redis-app
spec:
  containers:
  - name: app
    image: redis
    ports:
    - containerPort: 6379
"""

_POD_NGINX_INIT = """apiVersion: v1
kind: Pod
metadata:
  name: nginx-with-init
spec:
  initContainers:
  - name: setup
    image: busybox
    command: ["sh", "-c", "echo init"]
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_POD_REDIS_INIT = """apiVersion: v1
kind: Pod
metadata:
  name: redis-with-init
spec:
  initContainers:
  - name: setup
    image: busybox
    command: ["sh", "-c", "echo init"]
  containers:
  - name: app
    image: redis
    ports:
    - containerPort: 6379
"""

_SVC_CLUSTERIP_WEB = """apiVersion: v1
kind: Service
metadata:
  name: web-clusterip
spec:
  type: ClusterIP
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080
"""

_SVC_CLUSTERIP_API = """apiVersion: v1
kind: Service
metadata:
  name: api-clusterip
spec:
  type: ClusterIP
  selector:
    app: api
  ports:
  - port: 443
    targetPort: 8443
"""

_SVC_NODEPORT = """apiVersion: v1
kind: Service
metadata:
  name: web-nodeport
spec:
  type: NodePort
  selector:
    app: web
  ports:
  - port: 80
    targetPort: 8080
    nodePort: 30080
"""

_SVC_LOADBALANCER = """apiVersion: v1
kind: Service
metadata:
  name: web-lb
spec:
  type: LoadBalancer
  selector:
    app: web
  externalTrafficPolicy: Local
  ports:
  - port: 80
    targetPort: 8080
"""

_DEPLOY_NGINX = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx
spec:
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
      - name: app
        image: nginx
        ports:
        - containerPort: 80
"""

_CONFIGMAP_APP = """apiVersion: v1
kind: ConfigMap
metadata:
  name: app-config
data:
  config.yaml: |
    debug: true
  app.properties: |
    key1=value1
"""

_POD_NS_PROD_1 = """apiVersion: v1
kind: Pod
metadata:
  name: web-1
  namespace: production
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_POD_NS_PROD_2 = """apiVersion: v1
kind: Pod
metadata:
  name: web-2
  namespace: production
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_POD_NS_STAGING_1 = """apiVersion: v1
kind: Pod
metadata:
  name: web-1
  namespace: staging
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_POD_NS_STAGING_2 = """apiVersion: v1
kind: Pod
metadata:
  name: web-2
  namespace: staging
spec:
  containers:
  - name: app
    image: nginx
    ports:
    - containerPort: 80
"""

_DEPLOY_APPS_V1 = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: app
        image: nginx
        ports:
        - containerPort: 80
"""

_DEPLOY_EXT_V1BETA1 = """apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: app
        image: nginx
        ports:
        - containerPort: 80
"""

_REPLICASET_APPS_V1 = """apiVersion: apps/v1
kind: ReplicaSet
metadata:
  name: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: app
        image: nginx
        ports:
        - containerPort: 80
"""

_CONFIGMAP_PLAIN = """apiVersion: v1
kind: ConfigMap
metadata:
  name: web
data:
  config.yaml: |
    debug: true
  app.properties: |
    key=value
"""


def _verdict_init(cos):
    cd = float(cos[2][3])
    mx = float(max(cos[0][2], cos[1][3]))
    passed = cd > mx
    msg = (
        f"`cos(C, D)` = **{cd:.3f}** (both have init) · "
        f"`max(cos(A,C), cos(B,D))` = **{mx:.3f}** (mixed). "
        + ("Init-pairs cluster tighter than mixed pairs — the model treats "
           "`initContainers` as a real structural feature, stronger than the "
           "value-content similarity (shared `image: nginx` etc.)."
           if passed else
           "Mixed pairs are at least as close as init-pairs. This is not a "
           "regression — it reveals a re-balance: BPE makes `image` values "
           "compositionally visible to attention, so pods sharing `nginx` "
           "cluster together regardless of init presence. v8 with atomic "
           "`[UNK]` values had no choice but to lean on structure; v9 has "
           "both signals and now weights content more heavily here. "
           "Whether that's good depends on use case (good for content "
           "retrieval, less ideal for structure-only similarity).")
    )
    return passed, msg


def _verdict_service_type(cos):
    same = float(cos[0][1])
    cross = float(max(cos[0][2], cos[0][3], cos[1][2], cos[1][3], cos[2][3]))
    passed = same > cross
    msg = (
        f"`cos(A, B)` = **{same:.3f}** (both ClusterIP) · "
        f"`max(cross-type)` = **{cross:.3f}**. "
        + ("Same-type pairs cluster tighter than cross-type pairs — the "
           "model has internalized the `type` distinction (likely through "
           "the structural keys each type adds: `nodePort`, "
           "`externalTrafficPolicy`)."
           if passed else
           "Same-type pairs do not cluster more tightly than cross-type "
           "pairs — `type` is not a primary axis in the embedding here.")
    )
    return passed, msg


def _verdict_namespace(cos):
    # A, B in production · C, D in staging — all otherwise identical Pods.
    same_ns = float(min(cos[0][1], cos[2][3]))
    cross_ns = float(max(cos[0][2], cos[0][3], cos[1][2], cos[1][3]))
    passed = same_ns > cross_ns
    msg = (
        f"`min(same-ns cos)` = **{same_ns:.3f}** · "
        f"`max(cross-ns cos)` = **{cross_ns:.3f}**. "
        + ("Same-namespace pairs cluster tighter — value content reaches "
           "`doc_vec` even though the aggregator only sums KEY subtrees. "
           "BPE makes namespace values compositional (e.g., `prod | uction`), "
           "and self-attention spreads that signal into neighboring KEY "
           "hidden states, which then flow into `doc_vec`. This was the "
           "first failure of `[UNK]`-vocab v8 that v9 fixed."
           if passed else
           "Same-namespace and cross-namespace pairs have indistinguishable "
           "cosines. v8 saw this because both `production` and `staging` "
           "often hit `[UNK]`. v9 was expected to pass — if it does not, "
           "investigate.")
    )
    return passed, msg


def _verdict_apiversion(cos):
    # A = apps/v1 Deployment
    # B = extensions/v1beta1 Deployment  (same kind, deprecated apiVersion)
    # C = apps/v1 ReplicaSet              (different kind, same group, similar structure)
    # D = v1 ConfigMap                    (unrelated)
    same_kind = float(cos[0][1])
    same_group_diff_kind = float(cos[0][2])
    unrelated = float(cos[0][3])
    primary = same_kind > same_group_diff_kind
    secondary = same_group_diff_kind > unrelated
    passed = primary and secondary
    msg = (
        f"`cos(apps/v1 Dep, ext/v1beta1 Dep)` = **{same_kind:.3f}** · "
        f"`cos(Dep, ReplicaSet)` = **{same_group_diff_kind:.3f}** · "
        f"`cos(Dep, ConfigMap)` = **{unrelated:.3f}**. "
        + ("Same-kind-different-apiVersion pairs cluster tightest, then "
           "same-group-different-kind, then unrelated. The model treats "
           "apiVersion as a soft label, not a hard discriminator — it "
           "recognizes `apps/v1` and `extensions/v1beta1` Deployments as "
           "the same thing despite the apiVersion text differing."
           if passed else
           f"Expected ordering broken: same-kind={same_kind:.3f}, "
           f"same-group={same_group_diff_kind:.3f}, unrelated={unrelated:.3f}. "
           "If same-kind is NOT the tightest, the model is treating "
           "apiVersion as a hard discriminator and separating same-kind "
           "manifests by their api label — possibly an over-correction "
           "from the eval-probe accuracy.")
    )
    return passed, msg


def _verdict_cross_kind(cos):
    # A=Pod nginx, B=Pod redis, C=Deployment with nginx Pod template, D=ConfigMap
    pod_pod = float(cos[0][1])             # both Pods
    pod_deploy = float(cos[0][2])          # Pod ↔ Deployment with same-shape Pod template
    pod_cm = float(cos[0][3])              # Pod ↔ unrelated kind
    deploy_cm = float(cos[2][3])           # Deployment ↔ unrelated kind
    passed = pod_deploy > max(pod_cm, deploy_cm)
    msg = (
        f"`cos(Pod, Deployment-with-Pod-template)` = **{pod_deploy:.3f}** · "
        f"`cos(Pod, ConfigMap)` = **{pod_cm:.3f}** · "
        f"`cos(Pod, Pod)` = **{pod_pod:.3f}** (kind silo). "
        + ("Pod and the Deployment containing the same Pod template are "
           "closer than either is to an unrelated ConfigMap — the model "
           "encodes the shared Pod-shape across the two kinds."
           if passed else
           "The shared Pod-shape across the two kinds is not detected — "
           "kinds form sharp silos in the embedding.")
    )
    return passed, msg


PRESETS = [
    {
        "id": "init",
        "title": "Pod ± initContainers",
        "hypothesis": (
            "If the model treats `initContainers` as a structural feature, "
            "Pods that both have one should land closer in embedding space "
            "than mixed pairs. _Caveat: the init container in C and D is "
            "the same busybox setup, so `cos(C, D)` reflects both "
            "structural-key presence AND shared busybox content — this "
            "test is necessary but not sufficient for the structural "
            "claim. A follow-up varying the init-container content would "
            "isolate the two signals._"
        ),
        "manifests": [
            {"name": "nginx (no init)",   "yaml": _POD_NGINX},
            {"name": "redis (no init)",   "yaml": _POD_REDIS},
            {"name": "nginx + init",      "yaml": _POD_NGINX_INIT},
            {"name": "redis + init",      "yaml": _POD_REDIS_INIT},
        ],
        "verdict_fn": _verdict_init,
    },
    {
        "id": "service-type",
        "title": "Service type (ClusterIP / NodePort / LoadBalancer)",
        "hypothesis": (
            "Each Service `type` brings its own structural keys "
            "(`nodePort`, `externalTrafficPolicy`, …). If the model "
            "encodes `type` as a structural axis, same-type Services "
            "should sit closer in embedding space than cross-type ones, "
            "even when their selectors and ports differ."
        ),
        "manifests": [
            {"name": "ClusterIP — app=web",       "yaml": _SVC_CLUSTERIP_WEB},
            {"name": "ClusterIP — app=api",       "yaml": _SVC_CLUSTERIP_API},
            {"name": "NodePort — app=web",        "yaml": _SVC_NODEPORT},
            {"name": "LoadBalancer — app=web",    "yaml": _SVC_LOADBALANCER},
        ],
        "verdict_fn": _verdict_service_type,
    },
    {
        "id": "namespace",
        "title": "Pods in same namespace vs different namespace",
        "hypothesis": (
            "If the model encodes `metadata.namespace` as a feature, two "
            "Pods in the same namespace should be closer than two Pods in "
            "different namespaces (controlling for structure). "
            "_v8 with atomic vocab failed this probe — `production` and "
            "`staging` often mapped to `[UNK]`, leaving attention with no "
            "compositional content to work with. v9's byte-level BPE "
            "decomposes namespace values into subwords, and self-attention "
            "now spreads value content into surrounding KEY hidden states. "
            "Those KEYs are what the aggregator pools into `doc_vec` — so "
            "namespace effectively reaches `doc_vec` through the attention "
            "channel, even though the aggregator stays KEY-only by design._"
        ),
        "manifests": [
            {"name": "production / web-1", "yaml": _POD_NS_PROD_1},
            {"name": "production / web-2", "yaml": _POD_NS_PROD_2},
            {"name": "staging / web-1",    "yaml": _POD_NS_STAGING_1},
            {"name": "staging / web-2",    "yaml": _POD_NS_STAGING_2},
        ],
        "verdict_fn": _verdict_namespace,
    },
    {
        "id": "apiversion",
        "title": "apiVersion sensitivity (same kind, different apiVersion)",
        "hypothesis": (
            "K8s supports multiple `apiVersion`s for the same kind "
            "(e.g., `apps/v1` Deployment and the deprecated "
            "`extensions/v1beta1` Deployment have nearly identical "
            "structure). If the model encodes kind as the dominant "
            "structural signal and `apiVersion` as a soft label, two "
            "same-kind manifests with different `apiVersion`s should "
            "still sit closer in embedding space than a same-group "
            "different-kind manifest (`apps/v1` ReplicaSet), which in "
            "turn should be closer than an unrelated kind "
            "(`v1` ConfigMap). _Eval probes already show 99.8% "
            "`apiVersion` classification accuracy — but classification "
            "is compatible with either treating `apiVersion` as a soft "
            "label or as a hard discriminator. This probe tests which._"
        ),
        "manifests": [
            {"name": "apps/v1 Deployment",         "yaml": _DEPLOY_APPS_V1},
            {"name": "extensions/v1beta1 Deploy",  "yaml": _DEPLOY_EXT_V1BETA1},
            {"name": "apps/v1 ReplicaSet",         "yaml": _REPLICASET_APPS_V1},
            {"name": "v1 ConfigMap (unrelated)",   "yaml": _CONFIGMAP_PLAIN},
        ],
        "verdict_fn": _verdict_apiversion,
    },
    {
        "id": "cross-kind",
        "title": "Pod vs Deployment containing the same Pod template",
        "hypothesis": (
            "A Deployment's `spec.template` carries a Pod's shape — the "
            "same `spec.containers` substructure as a standalone Pod, "
            "just nested one level deeper. If the model encodes that "
            "shape similarity, a standalone Pod and a Deployment "
            "containing the same Pod template should sit closer in "
            "embedding space than either does to an unrelated kind like "
            "ConfigMap."
        ),
        "manifests": [
            {"name": "Pod (nginx)",                       "yaml": _POD_NGINX},
            {"name": "Pod (redis)",                       "yaml": _POD_REDIS},
            {"name": "Deployment with nginx template",    "yaml": _DEPLOY_NGINX},
            {"name": "ConfigMap (unrelated)",             "yaml": _CONFIGMAP_APP},
        ],
        "verdict_fn": _verdict_cross_kind,
    },
]

# Letters & colors used for both plot points and accordion headers.
_PRESET_PALETTE = [
    "#1f77b4",  # A blue
    "#ff7f0e",  # B orange
    "#2ca02c",  # C green
    "#d62728",  # D red
    "#9467bd",  # E purple
    "#8c564b",  # F brown
    "#e377c2",  # G pink
    "#17becf",  # H cyan
]
_PRESET_LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H"]


# Reuse one linearizer/annotator/config across calls (cheap, but skip the
# per-call construction inside _encode_doc_vec).
def _make_encoder_state():
    from yaml_bert.annotator import DomainAnnotator
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.linearizer import YamlLinearizer
    return YamlLinearizer(), DomainAnnotator(), YamlBertConfig(
        mask_prob=0.0, recon_enabled=False,
    )


_LINEARIZER, _ANNOTATOR, _INFER_CONFIG = _make_encoder_state()


def _encode_doc_vec(yaml_text: str) -> torch.Tensor:
    """Encode one YAML doc and return its doc_vec (shape (d_model,))."""
    from yaml_bert.dataset import YamlBertDataset, collate_fn as _collate
    nodes = _LINEARIZER.linearize(yaml_text)
    if not nodes:
        raise ValueError("YAML produced no nodes")
    _ANNOTATOR.annotate(nodes)
    ds = YamlBertDataset([nodes], VOCAB, _INFER_CONFIG)
    item = ds[0]
    batch = _collate([item])
    with torch.no_grad():
        out = MODEL(
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
    return out[1][0]  # (d_model,)


def _layout_2d(vecs):
    """Project doc_vecs to 2D coords via MDS on cosine distances."""
    import numpy as np
    n = len(vecs)
    if n == 1:
        return np.array([[0.0, 0.0]])
    if n == 2:
        return np.array([[-1.0, 0.0], [1.0, 0.0]])

    if isinstance(vecs, list):
        vecs = torch.stack(vecs)
    vecs_norm = vecs / vecs.norm(dim=1, keepdim=True)
    cos = (vecs_norm @ vecs_norm.t()).numpy()
    dist = 1.0 - cos
    np.fill_diagonal(dist, 0.0)
    dist = np.clip(dist, 0.0, None)
    # sklearn MDS(metric="precomputed") requires strict symmetry; floating-point
    # matmul can leave ~1e-7 asymmetries. Symmetrize explicitly.
    dist = (dist + dist.T) / 2

    from sklearn.manifold import MDS
    mds = MDS(
        n_components=2,
        metric="precomputed",
        random_state=42,
        normalized_stress="auto",
        init="classical_mds",
    )
    return mds.fit_transform(dist)


def _find_identical_groups(items, threshold: float = 0.9999):
    """Find groups of items whose pairwise cosine similarity is ~1.0.
    These represent inputs the model encodes to the same `doc_vec`
    (e.g. when their differing values all map to the same vocab token,
    typically [UNK]). The plot must show this honestly: they overlap.
    """
    if len(items) < 2:
        return []
    vecs = torch.stack([it["vec"] for it in items])
    vecs_norm = vecs / vecs.norm(dim=1, keepdim=True)
    cos = (vecs_norm @ vecs_norm.t()).numpy()

    groups: list[list[int]] = []
    assigned = [False] * len(items)
    for i in range(len(items)):
        if assigned[i]:
            continue
        group = [i]
        assigned[i] = True
        for j in range(i + 1, len(items)):
            if not assigned[j] and cos[i][j] >= threshold:
                group.append(j)
                assigned[j] = True
        if len(group) > 1:
            groups.append(group)
    return groups


def _collision_note(items):
    """Markdown note when two or more items produce identical doc_vecs."""
    groups = _find_identical_groups(items)
    if not groups:
        return ""
    lines = []
    for g in groups:
        labels = ", ".join(f"`{items[i]['letter']}`" for i in g)
        lines.append(
            f"- {labels} encode to **identical** `doc_vec`s "
            f"(`cos ≈ 1.0`). The plot will show them overlapping — that "
            f"is honest: the model cannot tell these inputs apart. "
            f"Likely cause: differing values all collapse to the same "
            f"vocab token (often `[UNK]` when names aren't frequent "
            f"enough in the training corpus to earn a vocab slot)."
        )
    return "\n\n**Identical embeddings detected:**\n" + "\n".join(lines)


def _build_preset_figure(items, hypothesis=""):
    """Build a Plotly scatter from a list of {letter, name, vec, ...} items."""
    import plotly.graph_objects as go
    if not items:
        return go.Figure()
    vecs = [it["vec"] for it in items]
    coords = _layout_2d(vecs)
    colors = [_PRESET_PALETTE[i % len(_PRESET_PALETTE)] for i in range(len(items))]
    letters = [it["letter"] for it in items]
    names = [it["name"] for it in items]

    fig = go.Figure()
    fig.add_scatter(
        x=coords[:, 0],
        y=coords[:, 1],
        mode="markers+text",
        text=letters,
        textposition="top center",
        textfont=dict(size=14, color="#222"),
        marker=dict(size=18, color=colors, opacity=0.85,
                    line=dict(width=2, color="#222")),
        hovertext=[f"<b>{l}</b> — {n}" for l, n in zip(letters, names)],
        hovertemplate="%{hovertext}<extra></extra>",
        showlegend=False,
    )
    fig.update_layout(
        title="2D layout (MDS of cosine distances) — closer = more similar",
        height=460,
        margin=dict(l=10, r=10, t=50, b=10),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="white",
    )
    return fig


def _encode_preset(preset):
    """Encode all manifests in a preset; return list of items with vecs."""
    items = []
    for i, m in enumerate(preset["manifests"]):
        vec = _encode_doc_vec(m["yaml"])
        items.append({
            "letter": _PRESET_LETTERS[i],
            "name": m["name"],
            "yaml": m["yaml"],
            "vec": vec,
            "preset_id": preset["id"],
        })
    return items


def _compute_cos_matrix(items):
    """Compute cosine matrix from items (for the verdict function)."""
    vecs = torch.stack([it["vec"] for it in items])
    vecs_norm = vecs / vecs.norm(dim=1, keepdim=True)
    return (vecs_norm @ vecs_norm.t()).numpy()


def _verdict_markdown(preset, items):
    """Run the preset's verdict function on the first N items (its presets)."""
    n_preset = len(preset["manifests"])
    if len(items) < n_preset:
        return "_(not enough manifests for verdict)_"
    cos = _compute_cos_matrix(items[:n_preset])
    passed, msg = preset["verdict_fn"](cos)
    emoji = "✅" if passed else "❌"
    main = f"**Verdict on preset:** {emoji} &nbsp; {msg}"
    return main + _collision_note(items)


_log("Encoding preset manifests...")
PRESET_BY_ID = {p["id"]: p for p in PRESETS}
PRESET_ITEMS_BY_ID = {p["id"]: _encode_preset(p) for p in PRESETS}
for p in PRESETS:
    items = PRESET_ITEMS_BY_ID[p["id"]]
    cos = _compute_cos_matrix(items)
    passed, _ = p["verdict_fn"](cos)
    _log(f"  '{p['title']}': {'PASS' if passed else 'FAIL'}")


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
_TILE_CSS = """
.demo-tile { display: flex !important; flex-direction: column !important; height: 100%; }
.demo-tile > .prose, .demo-tile > .markdown { flex: 1 1 auto; }
.demo-tile > button { margin-top: auto !important; }
"""
with gr.Blocks(title="YAML-BERT") as demo:
    gr.Markdown(
        f"""
# YAML-BERT — structural understanding of Kubernetes YAML

Code: [github.com/vimalk78/yaml-bert](https://github.com/vimalk78/yaml-bert) ·
Trained with MLM + reconstruction on 276K K8s manifests ·
{n_params:,} params

**This Space includes 3 demos — pick a tab below, or use the tiles on the Overview tab.**
"""
    )

    with gr.Tabs() as tabs:
        with gr.Tab("Overview", id="overview"):
            gr.Markdown("### Demos in this Space")
            with gr.Row():
                with gr.Column(elem_classes="demo-tile"):
                    gr.Markdown(
                        "#### 🧩 Missing-field suggester\n"
                        "Paste a YAML manifest; the model predicts which "
                        "structural fields it expects but you didn't include, "
                        "ranked by confidence."
                    )
                    open_suggester = gr.Button(
                        "Open missing-field suggester →", variant="primary"
                    )
                with gr.Column(elem_classes="demo-tile"):
                    gr.Markdown(
                        "#### 🌌 Manifest galaxy\n"
                        "10,000 K8s manifests projected to 2D from their "
                        "`doc_vec` embeddings. Kinds cluster spontaneously — "
                        "the model was never told what `kind` is."
                    )
                    open_galaxy = gr.Button(
                        "Open manifest galaxy →", variant="primary"
                    )
                with gr.Column(elem_classes="demo-tile"):
                    gr.Markdown(
                        "#### 🔬 Structural probes\n"
                        "Preset manifest sets that test whether the model "
                        "has learned specific structural distinctions, "
                        "shown as a 2D layout. Add your own YAML to see "
                        "where it lands."
                    )
                    open_probes = gr.Button(
                        "Open structural probes →", variant="primary"
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
                        minimum=0.05, maximum=0.95, value=0.7, step=0.05,
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

        with gr.Tab("Structural probes", id="probes"):
            gr.Markdown(
                "Pick a preset that explores a specific structural claim. "
                "The 2D plane is an MDS projection of the manifests' "
                "`doc_vecs` — **closer = more similar**. "
                "You can also paste your own YAML to see where it lands "
                "relative to the preset."
            )

            _initial_preset = PRESETS[0]
            _initial_items = PRESET_ITEMS_BY_ID[_initial_preset["id"]]

            preset_dd = gr.Dropdown(
                choices=[(p["title"], p["id"]) for p in PRESETS],
                value=_initial_preset["id"],
                label="Preset",
                interactive=True,
            )
            hypothesis_md = gr.Markdown(
                value=f"**Hypothesis:** {_initial_preset['hypothesis']}"
            )

            items_state = gr.State(value=_initial_items)

            with gr.Row():
                with gr.Column(scale=2):
                    plot = gr.Plot(
                        value=_build_preset_figure(_initial_items),
                        show_label=False,
                    )
                    verdict_md = gr.Markdown(
                        value=_verdict_markdown(_initial_preset, _initial_items)
                    )

                with gr.Column(scale=1):
                    gr.Markdown("**Manifests in this comparison**")

                    @gr.render(inputs=[items_state])
                    def _render_accordions(items):
                        if not items:
                            gr.Markdown("_No manifests._")
                            return
                        preset_id = items[0].get("preset_id", "")
                        n_preset = (len(PRESET_BY_ID[preset_id]["manifests"])
                                    if preset_id in PRESET_BY_ID else 0)
                        for i, it in enumerate(items):
                            is_user = i >= n_preset
                            label = f"[{it['letter']}] {it['name']}"
                            if is_user:
                                label += "  · added"
                            with gr.Accordion(label, open=False):
                                gr.Code(
                                    value=it["yaml"], language="yaml",
                                    lines=12, interactive=False,
                                )
                                if is_user:
                                    rm = gr.Button(
                                        f"Remove {it['letter']}",
                                        variant="secondary",
                                    )

                                    def _remove(state, idx=i):
                                        new = state[:idx] + state[idx + 1:]
                                        preset = PRESET_BY_ID.get(
                                            new[0]["preset_id"]) if new else None
                                        verdict = (_verdict_markdown(preset, new)
                                                   if preset else "")
                                        return (new,
                                                _build_preset_figure(new),
                                                verdict)
                                    rm.click(
                                        _remove,
                                        inputs=[items_state],
                                        outputs=[items_state, plot, verdict_md],
                                    )

                    gr.Markdown("---")
                    with gr.Accordion("➕ Add your own YAML", open=False):
                        new_yaml = gr.Code(
                            language="yaml", lines=10,
                            label="Paste a K8s manifest",
                        )
                        add_btn = gr.Button(
                            "Encode and add to comparison", variant="primary",
                        )
                        add_err = gr.Markdown("")

            def _on_add(state, yaml_text):
                if not yaml_text or not yaml_text.strip():
                    return state, _build_preset_figure(state), \
                        _verdict_markdown_for(state), \
                        "_Empty YAML — nothing to add._"
                if len(state) >= len(_PRESET_LETTERS):
                    return state, _build_preset_figure(state), \
                        _verdict_markdown_for(state), \
                        f"_Max {len(_PRESET_LETTERS)} manifests at once._"
                try:
                    vec = _encode_doc_vec(yaml_text)
                except Exception as e:
                    return state, _build_preset_figure(state), \
                        _verdict_markdown_for(state), \
                        f"_Encoding failed: `{type(e).__name__}: {e}`_"
                from yaml_bert.types import _extract_kind
                nodes = _LINEARIZER.linearize(yaml_text)
                kind = _extract_kind(nodes) or "?"
                preset_id = state[0]["preset_id"] if state else ""
                new_item = {
                    "letter": _PRESET_LETTERS[len(state)],
                    "name": f"{kind} (added)",
                    "yaml": yaml_text,
                    "vec": vec,
                    "preset_id": preset_id,
                }
                new_state = state + [new_item]
                return (new_state,
                        _build_preset_figure(new_state),
                        _verdict_markdown_for(new_state),
                        "")

            def _verdict_markdown_for(state):
                if not state:
                    return ""
                preset = PRESET_BY_ID.get(state[0].get("preset_id", ""))
                return _verdict_markdown(preset, state) if preset else ""

            def _on_preset_change(preset_id):
                if preset_id not in PRESET_BY_ID:
                    return gr.update(), gr.update(), gr.update(), gr.update()
                preset = PRESET_BY_ID[preset_id]
                items = PRESET_ITEMS_BY_ID[preset_id]
                return (items,
                        _build_preset_figure(items),
                        f"**Hypothesis:** {preset['hypothesis']}",
                        _verdict_markdown(preset, items))

            preset_dd.change(
                _on_preset_change,
                inputs=[preset_dd],
                outputs=[items_state, plot, hypothesis_md, verdict_md],
            )
            add_btn.click(
                _on_add,
                inputs=[items_state, new_yaml],
                outputs=[items_state, plot, verdict_md, add_err],
            )

    open_suggester.click(lambda: gr.Tabs(selected="suggester"), outputs=tabs)
    open_galaxy.click(lambda: gr.Tabs(selected="galaxy"), outputs=tabs)
    open_probes.click(lambda: gr.Tabs(selected="probes"), outputs=tabs)


_log("Gradio UI built — launching")

if __name__ == "__main__":
    # GRADIO_SHARE=true creates a temporary *.gradio.live tunnel.
    # Defaults off — Spaces deployment must stay local-bound.
    import os
    demo.launch(
        css=_TILE_CSS,
        share=os.environ.get("GRADIO_SHARE", "").lower() in ("1", "true", "yes"),
    )
