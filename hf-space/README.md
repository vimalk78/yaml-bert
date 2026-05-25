---
title: Yaml Bert
emoji: 💻
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.14.0
python_version: '3.13'
app_file: app.py
pinned: false
license: mit
short_description: Tree-aware transformer for structured (YAML) data
---

# YAML-BERT — Kubernetes missing-field suggester

A 22.5M-param BERT-style encoder trained on 276K Kubernetes YAML
manifests with **tree-aware positional encoding** and a **bottom-up
tree aggregator** producing a document-level vector (`doc_vec`). The
prediction head conditions on `[h_i ; doc_vec ; s_parent]` — token
hidden state, whole-document context, and the immediate parent subtree
— to predict the missing atomic key. Paste a YAML manifest below; the
model identifies fields it expects to see but that are absent, ranked
by confidence.

This Space runs the **v8 (MLM+reconstruction) checkpoint**. v8 retired
v7's hybrid bigram/trigram targets in favor of atomic prediction
conditioned on document context. The reconstruction objective (masking
random subtrees and reconstructing their key sets) further improves
calibration on ambiguous structural positions and tightens the geometric
quality of `doc_vec` (which downstream embedding apps will use).

v8 matches v7 on the canonical 93-test capability benchmark (93/93),
achieves a perfect 13/13 on bigger-boat tests (vs v7's 11/13), and
naturally covers status-key prediction (`status.replicas`,
`status.conditions`, etc.) without v7's explicit vocab-exemption lever.

## What the model does

- Predicts standard Kubernetes structural fields (`spec`, `replicas`,
  `containers.image`, etc.)
- Distinguishes kind-specific fields (`Deployment.replicas`,
  `Service.ports`)
- Predicts status-side fields (`status.replicas`, `status.conditions`,
  `status.loadBalancer.ingress`) — natural at full-corpus atomic vocab
- Calibrated confidence: strong on common patterns, less confident on
  ambiguous positions (the reconstruction objective specifically helps
  here)
- Produces a 256-dim `doc_vec` per manifest (encodes kind / apiVersion
  / GroupVersionKind at ~100% probe accuracy) — future apps in this
  Space will use this for similarity search and clustering

## Known limitations

- Novel CRD instances and very rare annotation keys may collapse to
  `[UNK]`
- Trained on `substratusai/the-stack-yaml-k8s`
- Occasionally over-confident on invalid YAML structures (model thinks
  it knows what should be at a wrong position)

## Links

- Code & docs: <https://github.com/vimalk78/yaml-bert>
- Evaluation results: <https://github.com/vimalk78/yaml-bert/blob/main/docs/evaluation-results.md>
- v8 design rationale: <https://github.com/vimalk78/yaml-bert/blob/main/docs/key-value-design-rationale.md>
- v8 276K scale-up results: <https://github.com/vimalk78/yaml-bert/blob/main/docs/v8-276K-scaleup-results.md>
