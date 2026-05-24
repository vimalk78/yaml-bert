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

A small (7.8M-param) BERT-style encoder trained on 276K Kubernetes YAML
manifests with **tree-aware positional encoding** and **hybrid bigram/
trigram prediction targets**. Paste a YAML manifest below; the model
identifies fields it expects to see but that are absent, ranked by
confidence.

This Space runs the **v6.1 checkpoint** — same architecture as v5, with
a 5-line bug-fix to selective masking. See the project repo for full
details.

## What the model does

- Predicts standard Kubernetes structural fields (`spec`, `replicas`,
  `containers.image`, etc.)
- Distinguishes kind-specific fields (`Deployment.replicas`,
  `Service.ports`)
- Calibrated confidence: strong on common patterns, weaker on ambiguous
  positions

## Known limitations

- Status-side fields not yet predicted (addressed in v6.2)
- Novel CRD instances and very rare annotation keys may collapse to
  `[UNK]`
- Trained on `substratusai/the-stack-yaml-k8s`

## Links

- Code & docs: <https://github.com/vimalk78/yaml-bert>
- Evaluation results: <https://github.com/vimalk78/yaml-bert/blob/main/docs/evaluation-results.md>
- v6 plan: <https://github.com/vimalk78/yaml-bert/blob/main/docs/v6-plan.md>
