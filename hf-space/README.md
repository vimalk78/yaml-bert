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

A 13.4M-param BERT-style encoder trained on 276K Kubernetes YAML
manifests with **tree-aware positional encoding** and **hybrid bigram/
trigram prediction targets**. Paste a YAML manifest below; the model
identifies fields it expects to see but that are absent, ranked by
confidence.

This Space runs the **v7 checkpoint** — same architecture as v6.1 plus
broader output vocabulary (Lever 5 depth cap, per-category min_freq
filtering, status-key vocab exemption). v7 fixes v6.1's main limitation:
status keys like `replicas`, `conditions`, and `currentMetrics` are now
in the target vocab, so the model can predict them when masked. See the
project repo for full details.

## What the model does

- Predicts standard Kubernetes structural fields (`spec`, `replicas`,
  `containers.image`, etc.)
- Distinguishes kind-specific fields (`Deployment.replicas`,
  `Service.ports`)
- Calibrated confidence: strong on common patterns, weaker on ambiguous
  positions

## Known limitations

- Novel CRD instances and very rare annotation keys may collapse to
  `[UNK]`
- Trained on `substratusai/the-stack-yaml-k8s`

## Links

- Code & docs: <https://github.com/vimalk78/yaml-bert>
- Evaluation results: <https://github.com/vimalk78/yaml-bert/blob/main/docs/evaluation-results.md>
- v6 plan: <https://github.com/vimalk78/yaml-bert/blob/main/docs/v6-plan.md>
