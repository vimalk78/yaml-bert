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

# YAML-BERT

A tree-aware BERT-style encoder trained on 276K Kubernetes YAML
manifests. Predicts missing structural fields and produces a
document-level vector (`doc_vec`) that encodes manifest structure and
content.

**18.4M params · subword BPE vocab (8,192) · 20 epochs on an L4 GPU**

## What this Space offers

Three tabs:

1. **Missing-field suggester** — paste a YAML manifest; the model
   walks each parent level, identifies fields it expects to see but
   that are absent, and ranks them by confidence.

2. **Manifest galaxy** — 10,000 manifests from the training corpus
   embedded as `doc_vec`s and projected to 2D with UMAP. Kinds cluster
   spontaneously — the model was never told what `kind` is.

3. **Structural probes** — hand-crafted manifest sets that test
   specific structural claims: Pod ± initContainers, Service type,
   namespace clustering, apiVersion sensitivity, Pod vs Deployment
   wrapping. The 2D layout uses MDS over cosine distances so closer =
   more similar. You can also paste your own YAML to see where it
   lands.

## Architecture (briefly)

- **Tree positional encoding**: each node's position is `depth` +
  `sibling_index` + `node_type`, summed into the input vector
- **Bottom-up tree aggregator**: combines per-key subtree vectors
  into `doc_vec`
- **Key Head**: predicts atomic keys from
  `[h_logical ; doc_vec ; s_parent]` — local context plus
  whole-document context plus immediate parent subtree
- **Unified byte-level BPE**: 8,192 subword tokens for both keys and
  values. Strings like `web-1`, `web-2`, `web-3` decompose into shared
  subwords so the model can both relate and distinguish them.

## What the model does well

- Predicts standard Kubernetes structural fields (`spec`, `replicas`,
  `containers.image`, etc.)
- Distinguishes kind-specific fields (`Deployment.replicas`,
  `Service.ports`)
- Predicts status-side fields (`status.replicas`,
  `status.conditions`, `status.loadBalancer.ingress`)
- Calibrated confidence: strong on common patterns, weaker on
  ambiguous positions

## Known limitations

- Occasionally over-confident on invalid YAML structures
- Foreign-key consistency (e.g., `volumeMounts[*].name` must match a
  defined `volumes[*].name`) is not learned — attention doesn't
  perform cross-position equality checks
- Trained on `substratusai/the-stack-yaml-k8s`; novel CRDs and rare
  annotations are seen but not deeply understood

## Links

- Code & docs: <https://github.com/vimalk78/yaml-bert>
- Results: <https://github.com/vimalk78/yaml-bert/blob/main/docs/v9-subword-results.md>
- Design rationale: <https://github.com/vimalk78/yaml-bert/blob/main/docs/key-value-design-rationale.md>
- Architecture: <https://github.com/vimalk78/yaml-bert/blob/main/docs/architecture.md>
