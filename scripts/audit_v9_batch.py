"""v9 audit: one batch through dataset → collate → model.forward, print all shapes.

Run: `PYTHONPATH=. python scripts/audit_v9_batch.py`
"""
from __future__ import annotations

import torch

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.config import YamlBertConfig
from yaml_bert.dataset import YamlBertDataset, collate_fn
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary, VocabBuilder

TOKENIZER_PATH = "output_v8_276K_recon_seed42/unified_bpe_8k.json"

YAMLS = [
    """apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
""",
    """apiVersion: v1
kind: Service
metadata:
  name: web
spec:
  type: ClusterIP
  ports:
  - port: 80
""",
]


def main():
    lin = YamlLinearizer()
    ann = DomainAnnotator()
    docs = []
    for y in YAMLS:
        nodes = lin.linearize(y)
        ann.annotate(nodes)
        docs.append(nodes)

    atv = VocabBuilder.build_atomic_target_vocab(docs, min_freq=1)
    vocab = Vocabulary.from_tokenizer_path(TOKENIZER_PATH, atomic_target_vocab=atv)

    cfg = YamlBertConfig(
        d_model=64, num_layers=2, num_heads=4, d_ff=128,
        max_depth=8, max_sibling=8, max_seq_len=128,
        mask_prob=0.15, recon_enabled=True,
    )
    ds = YamlBertDataset(docs, vocab, cfg)
    batch = collate_fn([ds[0], ds[1]])

    print("=" * 60)
    print("BATCH SHAPES")
    print("=" * 60)
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:25s}: tuple{tuple(v.shape)} {v.dtype}")
        elif isinstance(v, dict):
            print(f"  {k:25s}: dict of {len(v)} entries")
        else:
            print(f"  {k:25s}: {type(v).__name__} of {len(v) if hasattr(v, '__len__') else '?'}")

    emb = YamlBertEmbedding(config=cfg, subword_vocab_size=vocab.subword_vocab_size)
    model = YamlBertModel(
        config=cfg, embedding=emb,
        atomic_vocab_size=vocab.atomic_target_vocab_size,
    )

    out = model(
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
        subtree_mask=batch.get("subtree_mask"),
        subtree_roots_flat=batch.get("subtree_roots_flat"),
    )

    print()
    print("=" * 60)
    print("MODEL OUTPUTS")
    print("=" * 60)
    if len(out) == 2:
        logits, doc_vec = out
        recon = None
    else:
        logits, doc_vec, recon = out
    print(f"  logits:   {tuple(logits.shape)}  finite={torch.isfinite(logits).all().item()}")
    print(f"  doc_vec:  {tuple(doc_vec.shape)}  finite={torch.isfinite(doc_vec).all().item()}")
    if recon is not None:
        print(f"  recon:    {tuple(recon.shape)}  finite={torch.isfinite(recon).all().item()}")

    assert torch.isfinite(logits).all(), "non-finite logits"
    assert torch.isfinite(doc_vec).all(), "non-finite doc_vec"
    if recon is not None:
        assert torch.isfinite(recon).all(), "non-finite recon"

    print()
    print("AUDIT PASSED.")


if __name__ == "__main__":
    main()
