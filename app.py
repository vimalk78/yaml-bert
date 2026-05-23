"""YAML-BERT missing-field suggester — Gradio demo.

Paste a Kubernetes YAML manifest; the model identifies fields it expects
to see but that are absent. Runs the v6.1 checkpoint (Lever 1 — selective
masking applied during training).

Run locally:
    pip install gradio
    PYTHONPATH=. python app.py
"""
from __future__ import annotations

import os
import sys

import gradio as gr
import torch

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.suggest import suggest_missing_fields
from yaml_bert.vocab import Vocabulary


# ----- Model loading (once at startup) -----

DEFAULT_CHECKPOINT = "output_v6.1_lever1_only_seed42/checkpoints/yaml_bert_v4_epoch_30.pt"
DEFAULT_VOCAB = "output_v6.1_lever1_only_seed42/vocab.json"


def load_model(checkpoint_path: str, vocab_path: str) -> tuple[YamlBertModel, Vocabulary]:
    vocab = Vocabulary.load(vocab_path)
    config = YamlBertConfig()
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModel(
        config=config,
        embedding=emb,
        simple_vocab_size=vocab.simple_target_vocab_size,
        kind_vocab_size=vocab.kind_target_vocab_size,
    )
    cp = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(cp["model_state_dict"])
    model.eval()
    return model, vocab


checkpoint_path = os.environ.get("YAML_BERT_CHECKPOINT", DEFAULT_CHECKPOINT)
vocab_path = os.environ.get("YAML_BERT_VOCAB", DEFAULT_VOCAB)

print(f"Loading model from {checkpoint_path} (vocab: {vocab_path})", file=sys.stderr)
MODEL, VOCAB = load_model(checkpoint_path, vocab_path)
n_params = sum(p.numel() for p in MODEL.parameters())
print(f"Loaded {n_params:,} parameters", file=sys.stderr)


# ----- Inference -----

MAX_LINES = 300


def suggest(yaml_text: str, threshold: float, top_k: int) -> str:
    yaml_text = (yaml_text or "").strip()
    if not yaml_text:
        return "_Paste a YAML manifest above to see missing-field suggestions._"

    n_lines = len(yaml_text.splitlines())
    if n_lines > MAX_LINES:
        return (
            f"⚠️ **YAML too large for this demo** — {n_lines} lines (limit: {MAX_LINES}).\n\n"
            f"The model was trained on manifests up to ~512 linearized nodes. Large "
            f"manifests (cluster-dumped Pods with rich annotations/init containers, deep CRDs) "
            f"exceed that and inference becomes slow and unreliable.\n\n"
            f"Try a smaller manifest, or trim the verbose sections (annotations, env vars, "
            f"deeply nested probes)."
        )

    try:
        suggestions, _skipped = suggest_missing_fields(
            MODEL, VOCAB, yaml_text,
            threshold=threshold, top_k=top_k,
        )
    except Exception as e:
        return f"**Error parsing YAML:**\n```\n{e}\n```"

    if not suggestions:
        return ("_No suggestions above threshold. Model thinks this YAML is "
                "either complete, or it doesn't have strong opinions at this confidence level._")

    # Group by parent_path for readability
    by_parent: dict[str, list[dict]] = {}
    for s in suggestions:
        by_parent.setdefault(s.get("parent_path") or "(root)", []).append(s)

    blocks: list[str] = []
    for parent in sorted(by_parent.keys(), key=lambda p: -max(s["confidence"] for s in by_parent[p])):
        rows = ["| Missing key | Confidence | Strength |", "|---|---:|---|"]
        for s in sorted(by_parent[parent], key=lambda x: -x["confidence"]):
            conf = s["confidence"]
            strength = "**STRONG**" if conf >= 0.7 else ("MODERATE" if conf >= 0.5 else "weak")
            rows.append(f"| `{s['missing_key']}` | {conf:.1%} | {strength} |")
        blocks.append(f"### `{parent}`\n" + "\n".join(rows))

    return "\n\n".join(blocks)


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

with gr.Blocks(title="YAML-BERT — missing-field suggester") as demo:
    gr.Markdown(
        f"""
# YAML-BERT — Kubernetes missing-field suggester

A small (7.8M-param) BERT-style encoder trained on 276K Kubernetes YAML manifests with
**tree-aware positional encoding** and **hybrid bigram/trigram prediction targets**.
This page runs the **v6.1 checkpoint** — same architecture as v5, with a 5-line bug
fix to selective masking ([details](https://github.com/vimalk78/yaml-bert/blob/main/docs/evaluation-results.md#7-v61--lever-1-selective-masking)).

**How it works:** the model walks each parent level in your YAML, inserts a fake
`[MASK]` node, and reports the keys it expects to see there but that aren't present.
Confidence reflects how strongly the model "expects" each missing field.

Code: [github.com/vimalk78/yaml-bert](https://github.com/vimalk78/yaml-bert) ·
{n_params:,} params
"""
    )

    with gr.Row():
        with gr.Column(scale=1):
            yaml_input = gr.Code(
                language="yaml",
                lines=22,
                max_lines=100,        # grow up to ~100 lines, then scroll inside
                label="YAML input",
                value=EXAMPLE_NGINX,
            )
            with gr.Row():
                threshold = gr.Slider(
                    minimum=0.1, maximum=0.95, value=0.75, step=0.05,
                    label="Confidence threshold",
                )
                top_k = gr.Slider(
                    minimum=3, maximum=20, value=10, step=1,
                    label="Top-K predictions per position",
                )
            submit = gr.Button("Suggest missing fields", variant="primary")

        with gr.Column(scale=1):
            output = gr.Markdown(label="Suggestions", value="")

    submit.click(fn=suggest, inputs=[yaml_input, threshold, top_k], outputs=output)
    # No auto-trigger on yaml_input.change — typing/pasting a long YAML would
    # fire many inference requests and back up the queue. Button click only.

    gr.Examples(
        examples=[
            [EXAMPLE_NGINX, 0.75, 10],
            [EXAMPLE_INCOMPLETE_SERVICE, 0.75, 10],
            [EXAMPLE_CONFIGMAP, 0.75, 10],
        ],
        inputs=[yaml_input, threshold, top_k],
        label="Example YAMLs",
    )

    gr.Markdown(
        """
---

### What this model does well
- Predicts standard Kubernetes structural fields (`spec`, `replicas`, `containers.image`, etc.)
- Distinguishes kind-specific fields (`Deployment.replicas`, `Service.ports`)
- Calibrated confidence: strong on common patterns, weaker on ambiguous positions

### Known limitations
- **Status-side fields not yet predicted** (still being addressed in v6.2)
- Novel CRD instances and very rare annotation keys may collapse to `[UNK]`
- Trained on `substratusai/the-stack-yaml-k8s` (276K manifests from HF)
"""
    )


if __name__ == "__main__":
    demo.launch()
