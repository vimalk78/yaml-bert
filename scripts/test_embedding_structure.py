"""Test whether learned tree positional encodings have mathematical structure.

Unlike sine/cosine positional encoding which has built-in structure (nearby
positions are similar), our depth/sibling/node_type embeddings start random
and are learned. This test checks what structure emerged from training.

Usage:
    PYTHONPATH=. python scripts/test_embedding_structure.py output_v4/checkpoints/yaml_bert_v4_epoch_15.pt --vocab output_v4/vocab.json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import torch
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v4/vocab.json")
    args = parser.parse_args()

    torch.manual_seed(42)
    vocab = Vocabulary.load(args.vocab)
    config = YamlBertConfig()
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size,
                            value_vocab_size=vocab.value_vocab_size)
    model = YamlBertModel(config=config, embedding=emb,
                          simple_vocab_size=vocab.simple_target_vocab_size,
                          kind_vocab_size=vocab.kind_target_vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded epoch {checkpoint['epoch']}\n")

    depth_emb = model.embedding.depth_embedding.weight.data
    sibling_emb = model.embedding.sibling_embedding.weight.data
    node_type_emb = model.embedding.node_type_embedding.weight.data

    # ================================================================
    print("=" * 60)
    print("  Depth Embedding Structure")
    print("=" * 60)
    print("\n  Do nearby depths have similar embeddings?")
    print(f"  {'Pair':<20} {'Cosine Sim':>10}")
    print(f"  {'-'*20} {'-'*10}")

    for i in range(min(10, config.max_depth - 1)):
        sim = cosine_sim(depth_emb[i], depth_emb[i + 1])
        print(f"  depth {i} vs {i+1}       {sim:>10.4f}")

    print(f"\n  Distant depths:")
    for i, j in [(0, 5), (0, 10), (1, 8), (2, 7)]:
        if j < config.max_depth:
            sim = cosine_sim(depth_emb[i], depth_emb[j])
            print(f"  depth {i} vs {j}       {sim:>10.4f}")

    # Check if there's a gradient: avg similarity for distance 1 vs distance 5
    near_sims = [cosine_sim(depth_emb[i], depth_emb[i+1]) for i in range(8)]
    far_sims = [cosine_sim(depth_emb[i], depth_emb[i+5]) for i in range(5)]
    print(f"\n  Average similarity — adjacent depths: {sum(near_sims)/len(near_sims):.4f}")
    print(f"  Average similarity — 5 apart:          {sum(far_sims)/len(far_sims):.4f}")
    if sum(near_sims)/len(near_sims) > sum(far_sims)/len(far_sims):
        print(f"  --> Nearby depths ARE more similar (structure learned)")
    else:
        print(f"  --> No distance gradient (no smooth structure)")

    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Sibling Embedding Structure")
    print("=" * 60)
    print("\n  Do nearby siblings have similar embeddings?")
    print(f"  {'Pair':<20} {'Cosine Sim':>10}")
    print(f"  {'-'*20} {'-'*10}")

    for i in range(min(8, config.max_sibling - 1)):
        sim = cosine_sim(sibling_emb[i], sibling_emb[i + 1])
        print(f"  sib {i} vs {i+1}         {sim:>10.4f}")

    near_sims = [cosine_sim(sibling_emb[i], sibling_emb[i+1]) for i in range(8)]
    far_sims = [cosine_sim(sibling_emb[i], sibling_emb[i+5]) for i in range(5)]
    print(f"\n  Average similarity — adjacent siblings: {sum(near_sims)/len(near_sims):.4f}")
    print(f"  Average similarity — 5 apart:            {sum(far_sims)/len(far_sims):.4f}")
    if sum(near_sims)/len(near_sims) > sum(far_sims)/len(far_sims):
        print(f"  --> Nearby siblings ARE more similar (ordering learned)")
    else:
        print(f"  --> No distance gradient (no smooth ordering)")

    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Node Type Embedding Structure")
    print("=" * 60)
    print("\n  Node types: KEY=0, VALUE=1, LIST_KEY=2, LIST_VALUE=3")
    print(f"  {'Pair':<25} {'Cosine Sim':>10}")
    print(f"  {'-'*25} {'-'*10}")

    names = ["KEY", "VALUE", "LIST_KEY", "LIST_VALUE"]
    for i in range(4):
        for j in range(i + 1, 4):
            sim = cosine_sim(node_type_emb[i], node_type_emb[j])
            print(f"  {names[i]:<10} vs {names[j]:<10} {sim:>10.4f}")

    key_sim = cosine_sim(node_type_emb[0], node_type_emb[2])
    val_sim = cosine_sim(node_type_emb[1], node_type_emb[3])
    cross_sim = cosine_sim(node_type_emb[0], node_type_emb[1])

    print(f"\n  KEY vs LIST_KEY:     {key_sim:.4f}  (should be high — both are keys)")
    print(f"  VALUE vs LIST_VALUE: {val_sim:.4f}  (should be high — both are values)")
    print(f"  KEY vs VALUE:        {cross_sim:.4f}  (should be low — different roles)")

    if key_sim > cross_sim and val_sim > cross_sim:
        print(f"  --> Key types cluster together, value types cluster together (structure learned)")
    else:
        print(f"  --> No clear key/value clustering")

    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Embedding Norms")
    print("=" * 60)
    print(f"\n  Do frequently used embeddings have larger norms?")
    print(f"  (Depths 0-3 are common, depths 10+ are rare)")
    print(f"  {'Embedding':<15} {'Norm':>8}")
    print(f"  {'-'*15} {'-'*8}")
    for i in range(min(12, config.max_depth)):
        norm = depth_emb[i].norm().item()
        print(f"  depth {i:<8} {norm:>8.2f}")

    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Kind Embedding Norms (via value embedding table)")
    print("=" * 60)
    val_emb = model.embedding.value_embedding.weight.data
    unk_id = vocab.special_tokens["[UNK]"]

    known_kinds: list[tuple[str, float]] = []
    unk_kinds: list[str] = []
    case_variants: dict[str, list[str]] = {}

    for kind_name in sorted(vocab.kind_vocab.keys()):
        if kind_name == "[NO_KIND]":
            continue
        val_id = vocab.encode_value(kind_name)
        if val_id == unk_id:
            unk_kinds.append(kind_name)
        else:
            norm = val_emb[val_id].norm().item()
            known_kinds.append((kind_name, norm))
        # Track case variants
        lower = kind_name.lower()
        case_variants.setdefault(lower, []).append(kind_name)

    known_kinds.sort(key=lambda x: -x[1])
    print(f"\n  Known kinds (have their own embedding):")
    print(f"  {'Kind':<35} {'Norm':>8}")
    print(f"  {'-'*35} {'-'*8}")
    for name, norm in known_kinds:
        print(f"  {name:<35} {norm:>8.2f}")

    print(f"\n  Unknown kinds (all map to [UNK], share same embedding):")
    print(f"  {', '.join(unk_kinds[:15])}")
    if len(unk_kinds) > 15:
        print(f"  ... and {len(unk_kinds) - 15} more")
    print(f"  Total: {len(unk_kinds)} kinds map to [UNK]")

    # ================================================================
    print(f"\n{'=' * 60}")
    print("  Case Sensitivity Issues")
    print("=" * 60)
    print(f"\n  Kind values with multiple case variants in the vocabulary:")
    has_variants = False
    for lower, variants in sorted(case_variants.items()):
        if len(variants) > 1:
            has_variants = True
            ids = [vocab.encode_value(v) for v in variants]
            print(f"  {' / '.join(variants)}")
            print(f"    value IDs: {ids} — {'SAME embedding' if len(set(ids)) == 1 else 'DIFFERENT embeddings'}")
    if not has_variants:
        print(f"  None found.")


if __name__ == "__main__":
    main()
