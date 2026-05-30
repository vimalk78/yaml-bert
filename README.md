# YAML-BERT

A tree-aware BERT-style encoder for Kubernetes YAML manifests. Produces
per-token predictions (missing-field suggestions) and a document-level
embedding (`doc_vec`) suitable for retrieval, clustering, and structural
probing.

**Try it:** [vimalk78/yaml-bert on Hugging Face Spaces](https://huggingface.co/spaces/vimalk78/yaml-bert) —
missing-field suggester, manifest galaxy (10K manifests projected to 2D),
and 5 structural probes.

## What This Is

A research project exploring how transformers learn from structured (tree)
data instead of sequential text. Three architectural ideas:

1. **Tree positional encoding** — each node's position is its `depth`,
   `sibling_index`, and `node_type`, summed into the input vector. YAML
   is a tree, not a sequence; the model is told so.

2. **Bottom-up tree aggregator** — combines per-key subtree vectors into
   a single document vector. The Key Head predicts missing keys
   conditioned on `[h_logical ; doc_vec ; s_parent]` so each prediction
   sees both the local context and whole-document context.

3. **Unified byte-level BPE subword vocabulary** — one 8,192-token
   vocab covers both keys and values. Strings like `web-1`, `web-2`,
   `web-3` decompose into shared subwords plus a distinguishing
   suffix, so the model can both relate them and tell them apart.

The model learns from 276K real Kubernetes YAMLs which structural keys
belong where, and produces an embedding that respects both structure
(kinds cluster spontaneously) and content (namespace values reach
`doc_vec` via attention).

## Results

Trained on 276K manifests from [substratusai/the-stack-yaml-k8s](https://huggingface.co/datasets/substratusai/the-stack-yaml-k8s).

| Metric | Value |
|---|---|
| Pre-training capability tests | 92/93 (27/28 capabilities) |
| Fine-tuning capability tests | 24/28 (calibration edge cases) |
| Structural tests | 8/9 (missing-metadata is a known gap) |
| Bigger-boat tests | 13/13 (100%) |
| Kind k-NN purity @ 5 | 98.9% |
| Model parameters | 18.4M |
| Training | 20 epochs, ~12 hrs, ~$6 on JarvisLabs L4 |
| Subword vocab | 8,192 (byte-level BPE) |
| Atomic target vocab | 11,080 keys (Key Head output) |

See [`docs/v9-subword-results.md`](docs/v9-subword-results.md) for the
full evaluation.

## Architecture

```
YAML manifest
    ↓
Linearizer → DFS-ordered list of YamlNode(token, depth, sibling, type, path)
    ↓
BPE Tokenizer → expand each node's token to subword id(s)
    ↓
Subword Embedding + Tree Positional Encoding (depth + sibling + node_type)
    ↓
Transformer Encoder (6 layers, 8 heads, d_model=256)
    ↓
Subword Pooling → mean-pool subwords per logical node
    ↓
Tree Aggregator → bottom-up combine of KEY subtree vectors → doc_vec
    ↓
┌────────────────────────────┬────────────────────────────────┐
↓                            ↓                                ↓
Key Head                     Reconstruction Head              doc_vec
[h_logical ; doc_vec ;       BCE over bag-of-keys             256-dim
 s_parent] → atomic key      for masked subtrees              (retrieval,
                                                              clustering)
```

### Key/value asymmetry

- **KEYs** are aggregated into `doc_vec`, masked in MLM, and predicted
  by the Key Head.
- **VALUEs** are not aggregated, never masked, never predicted — but
  their BPE subwords are visible to self-attention, which lets value
  content influence neighboring KEY hidden states (and therefore reach
  `doc_vec` indirectly). See
  [`docs/key-value-design-rationale.md`](docs/key-value-design-rationale.md).

## Usage

### Train

```bash
# Quick local check (5K docs, 3 epochs)
PYTHONPATH=. python scripts/train.py \
    --max-docs 5000 --epochs 3 --batch-size 8 \
    --reconstruction on --output-dir output_quick

# Full training (276K docs, 20 epochs)
PYTHONPATH=. python scripts/train.py \
    --max-docs 0 --epochs 20 --batch-size 32 \
    --reconstruction on --output-dir output_276K --seed 42
```

The training script auto-downloads the dataset and rebuilds the BPE
tokenizer cache if missing.

### Run evaluations

```bash
PYTHONPATH=. python model_tests/test_capabilities.py <checkpoint> --vocab <vocab>
PYTHONPATH=. python model_tests/test_structural.py    <checkpoint> --vocab <vocab>
PYTHONPATH=. python model_tests/test_bigger_boat.py   <checkpoint> --vocab <vocab>
PYTHONPATH=. python scripts/eval_probes.py            --output-dir <output_dir>
PYTHONPATH=. python scripts/v9_structural_probes.py   --checkpoint <ckpt> --vocab <vocab>
```

### Suggest missing fields

Paste a YAML manifest into the [HF Space](https://huggingface.co/spaces/vimalk78/yaml-bert),
or run the audit script locally:

```bash
PYTHONPATH=. python scripts/audit_v9_batch.py
```

## Project Structure

```
yaml_bert/              # Core library
  types.py              # YamlNode, NodeType
  linearizer.py         # YAML → linearized tree nodes
  annotator.py          # Domain annotations (list ordering)
  tokenizer.py          # SubwordTokenizer (wraps HF tokenizers)
  vocab.py              # Vocabulary + atomic_target_vocab
  config.py             # Hyperparameters (d_model=256, layers=6, heads=8, max_seq_len=768)
  embedding.py          # Subword embedding + tree positional encoding
  dataset.py            # BPE-expansion + whole-word MLM masking
  aggregator.py         # Subword pooling + bottom-up tree aggregation
  model.py              # YamlBertModel: encoder + aggregator + Key Head + Recon Head
  reconstruction_head.py
  subtree_masking.py
  suggest.py            # Missing-field suggestion
  cache.py              # Corpus linearization cache
  # Dormant building blocks (kept for future experiments):
  tree_bias.py          # Tree-distance attention bias (disabled for perf)
  pooling.py            # Cross-attention pooling + supervised contrastive loss
  attention_pooling.py  # Simple learned-attention pooling baseline

scripts/
  train.py                       # Main training script
  train_unified_tokenizer.py     # Offline BPE training
  audit_v9_batch.py              # Single-batch shape audit
  v9_structural_probes.py        # 5 HF-Space-style probes
  v10_failing_probes.py          # 3 probes designed to fail (future targets)
  eval_probes.py                 # sklearn probes on per-epoch doc_vec dumps
  build_galaxy.py                # UMAP projection for HF Space galaxy
  deploy_hf_space.sh             # Bundle + upload to HF Space

model_tests/
  test_capabilities.py     # 121+ tests across 30 capabilities
  test_structural.py       # 9 structural reasoning tests
  test_bigger_boat.py      # 13 tests for cross-kind generalization

hf-space/                  # HF Space deployable (Gradio app)
tests/                     # Unit tests (pytest)
testdata/                  # Sample K8s YAMLs
docs/                      # Documentation + design specs + plans
tokenizers/                # Trained BPE artifacts
```

## Documentation

- [Results](docs/v9-subword-results.md) — current training run and evaluation
- [Architecture](docs/architecture.md) — encoder, aggregator, prediction heads
- [Key/Value Design Rationale](docs/key-value-design-rationale.md) — why KEYs are first-class
- [Future Directions](docs/future-directions.md) — what's next
- [Tree Positional Encoding Explained](docs/tree-positional-encoding-explained.md) — mathematical foundations
- [Inference Pipeline](docs/inference-pipeline.md) — how missing-field suggestion works
- [Development plans](docs/superpowers/) — AI-assisted spec + plan docs (built with Claude Code)

## Requirements

```bash
pip install -r requirements.txt
```

- Python 3.10+
- PyTorch >= 2.0
- HuggingFace `tokenizers`, `datasets`
- PyYAML, tqdm

## Built With

This project was built with [Claude Code](https://claude.ai/code) as an
AI-assisted development experiment. The
[development plans](docs/superpowers/) show how AI was used for
architecture design, planning, and implementation.

## License

MIT — see [LICENSE](LICENSE).
