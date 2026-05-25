"""Verify mathematical claims made in docs/tree-positional-encoding-explained.md.

Tests three properties of tree positional encoding:
1. Uniqueness — distinct (depth, sibling, type) combos produce distinct vectors
2. Distance-sensitivity — more shared components = higher similarity
3. Decomposability — attention heads show structural specialization

Usage:
    PYTHONPATH=. python scripts/test_tpe_claims.py <checkpoint> --vocab <vocab>
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import itertools

import torch
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary
from yaml_bert.types import NodeType


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def test_uniqueness(depth_emb: torch.Tensor, sibling_emb: torch.Tensor,
                    node_type_emb: torch.Tensor, max_depth: int, max_sibling: int) -> None:
    """Claim 1: Every distinct (depth, sibling, type) produces a distinct TPE vector."""
    print("=" * 60)
    print("  Claim 1: UNIQUENESS")
    print("  Every distinct (depth, sibling, type) combination")
    print("  should produce a distinct TPE vector.")
    print("=" * 60)

    # Generate all TPE vectors for reasonable ranges
    test_depths = min(max_depth, 10)
    test_siblings = min(max_sibling, 8)
    num_types = 4

    vectors: dict[tuple[int, int, int], torch.Tensor] = {}
    for d, s, t in itertools.product(range(test_depths), range(test_siblings), range(num_types)):
        tpe = depth_emb[d] + sibling_emb[s] + node_type_emb[t]
        vectors[(d, s, t)] = tpe

    total_combos = len(vectors)
    print(f"\n  Generated {total_combos} TPE vectors ({test_depths} depths x {test_siblings} siblings x {num_types} types)")

    # Check for duplicates (cosine similarity == 1.0)
    duplicates: list[tuple] = []
    keys = list(vectors.keys())
    # Check a sample of pairs (all pairs would be O(n^2) = 102K comparisons)
    checked = 0
    near_duplicates = 0
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            sim = cosine_sim(vectors[keys[i]], vectors[keys[j]])
            checked += 1
            if sim > 0.99:
                near_duplicates += 1
                duplicates.append((keys[i], keys[j], sim))

    print(f"  Checked {checked:,} pairs")
    if near_duplicates == 0:
        print(f"  PASS: All {total_combos} vectors are distinct (no cosine similarity > 0.99)")
    else:
        print(f"  FAIL: Found {near_duplicates} near-duplicate pairs (cosine > 0.99):")
        for a, b, sim in duplicates[:5]:
            print(f"    {a} vs {b}: cosine = {sim:.6f}")

    # Also check linear independence of base embeddings
    print(f"\n  Linear independence of depth embeddings (first {test_depths}):")
    depth_matrix = depth_emb[:test_depths]
    rank = torch.linalg.matrix_rank(depth_matrix).item()
    print(f"    Matrix rank: {rank} / {test_depths} — {'FULL RANK (independent)' if rank == test_depths else 'RANK DEFICIENT'}")

    print(f"\n  Linear independence of sibling embeddings (first {test_siblings}):")
    sib_matrix = sibling_emb[:test_siblings]
    rank = torch.linalg.matrix_rank(sib_matrix).item()
    print(f"    Matrix rank: {rank} / {test_siblings} — {'FULL RANK (independent)' if rank == test_siblings else 'RANK DEFICIENT'}")

    print(f"\n  Linear independence of node type embeddings (all 4):")
    rank = torch.linalg.matrix_rank(node_type_emb).item()
    print(f"    Matrix rank: {rank} / 4 — {'FULL RANK (independent)' if rank == 4 else 'RANK DEFICIENT'}")


def test_distance_sensitivity(depth_emb: torch.Tensor, sibling_emb: torch.Tensor,
                               node_type_emb: torch.Tensor) -> None:
    """Claim 2: More shared components = higher similarity."""
    print(f"\n{'=' * 60}")
    print("  Claim 2: DISTANCE-SENSITIVITY")
    print("  Nodes sharing more TPE components should have higher")
    print("  cosine similarity than nodes sharing fewer components.")
    print("=" * 60)

    type_key = node_type_emb[0]
    type_val = node_type_emb[1]

    # 3 shared (identical)
    tpe_same = depth_emb[2] + sibling_emb[0] + type_key
    sim_3 = cosine_sim(tpe_same, tpe_same)

    # 2 shared: vary each component individually
    sim_2_vary_sib = cosine_sim(
        depth_emb[2] + sibling_emb[0] + type_key,
        depth_emb[2] + sibling_emb[1] + type_key,
    )
    sim_2_vary_depth = cosine_sim(
        depth_emb[1] + sibling_emb[0] + type_key,
        depth_emb[4] + sibling_emb[0] + type_key,
    )
    sim_2_vary_type = cosine_sim(
        depth_emb[2] + sibling_emb[0] + type_key,
        depth_emb[2] + sibling_emb[0] + type_val,
    )

    # 1 shared: vary two components
    sim_1_share_depth = cosine_sim(
        depth_emb[2] + sibling_emb[0] + type_key,
        depth_emb[2] + sibling_emb[3] + type_val,
    )
    sim_1_share_sib = cosine_sim(
        depth_emb[1] + sibling_emb[0] + type_key,
        depth_emb[4] + sibling_emb[0] + type_val,
    )
    sim_1_share_type = cosine_sim(
        depth_emb[1] + sibling_emb[0] + type_key,
        depth_emb[4] + sibling_emb[3] + type_key,
    )

    # 0 shared: all different
    sim_0 = cosine_sim(
        depth_emb[0] + sibling_emb[0] + type_key,
        depth_emb[5] + sibling_emb[4] + type_val,
    )

    avg_2 = (sim_2_vary_sib + sim_2_vary_depth + sim_2_vary_type) / 3
    avg_1 = (sim_1_share_depth + sim_1_share_sib + sim_1_share_type) / 3

    print(f"\n  {'Shared components':<30} {'Pair description':<40} {'Cosine':>8}")
    print(f"  {'-'*30} {'-'*40} {'-'*8}")
    print(f"  {'3/3 (identical)':<30} {'same depth, sib, type':<40} {sim_3:>8.4f}")
    print(f"  {'2/3 (vary sibling)':<30} {'depth=2, KEY, sib 0 vs 1':<40} {sim_2_vary_sib:>8.4f}")
    print(f"  {'2/3 (vary depth)':<30} {'sib=0, KEY, depth 1 vs 4':<40} {sim_2_vary_depth:>8.4f}")
    print(f"  {'2/3 (vary type)':<30} {'depth=2, sib=0, KEY vs VALUE':<40} {sim_2_vary_type:>8.4f}")
    print(f"  {'1/3 (share depth)':<30} {'depth=2, rest different':<40} {sim_1_share_depth:>8.4f}")
    print(f"  {'1/3 (share sibling)':<30} {'sib=0, rest different':<40} {sim_1_share_sib:>8.4f}")
    print(f"  {'1/3 (share type)':<30} {'KEY, rest different':<40} {sim_1_share_type:>8.4f}")
    print(f"  {'0/3 (all different)':<30} {'depth/sib/type all differ':<40} {sim_0:>8.4f}")

    print(f"\n  Averages:")
    print(f"    3 shared: {sim_3:.4f}")
    print(f"    2 shared: {avg_2:.4f}")
    print(f"    1 shared: {avg_1:.4f}")
    print(f"    0 shared: {sim_0:.4f}")

    if sim_3 > avg_2 > avg_1 > sim_0:
        print(f"\n  PASS: Monotonic decrease — 3 > 2 > 1 > 0 shared components")
    elif avg_2 > sim_0:
        print(f"\n  PARTIAL: 2 shared > 0 shared, but not strictly monotonic across all levels")
    else:
        print(f"\n  FAIL: No clear distance-sensitivity gradient")


def test_decomposability(model: YamlBertModel, vocab: Vocabulary) -> None:
    """Claim 3: Attention heads can specialize on structural relationships."""
    print(f"\n{'=' * 60}")
    print("  Claim 3: DECOMPOSABILITY IN ATTENTION")
    print("  Different attention heads should show specialization —")
    print("  some heads attend more within same depth, others within")
    print("  same node type, etc.")
    print("=" * 60)

    # Use a real YAML to get meaningful attention patterns
    yaml_text = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: test
  labels:
    app: test
spec:
  replicas: 3
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        ports:
        - containerPort: 80
"""
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_text)

    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
    token_ids, node_types, depths, siblings = [], [], [], []

    for node in nodes:
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(type_map[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))

    t = lambda x: torch.tensor([x])
    model.eval()

    # Extract attention weights from all layers
    x = model.embedding(t(token_ids), t(node_types), t(depths), t(siblings))
    seq_len = x.shape[1]

    all_attn: list[torch.Tensor] = []
    for layer in model.encoder.layers:
        _, attn_weights = layer.self_attn(x, x, x, need_weights=True, average_attn_weights=False)
        all_attn.append(attn_weights.detach())
        x = layer(x)

    depths_t = torch.tensor(depths)
    types_t = torch.tensor(node_types)

    print(f"\n  Sequence length: {seq_len} tokens")
    print(f"  Model: {len(model.encoder.layers)} layers, {all_attn[0].shape[1]} heads")

    print(f"\n  Per-head structural bias (how much more a head attends to")
    print(f"  same-depth or same-type pairs vs different pairs):")
    print(f"\n  {'Layer':<8} {'Head':<6} {'Same-depth bias':>16} {'Same-type bias':>16} {'Specialization':>16}")
    print(f"  {'-'*8} {'-'*6} {'-'*16} {'-'*16} {'-'*16}")

    for layer_idx, attn in enumerate(all_attn):
        num_heads = attn.shape[1]
        for head_idx in range(num_heads):
            head_attn = attn[0, head_idx]  # (seq, seq)

            # Compute average attention for same-depth vs different-depth pairs
            same_depth_mask = depths_t.unsqueeze(0) == depths_t.unsqueeze(1)
            diff_depth_mask = ~same_depth_mask

            same_depth_attn = head_attn[same_depth_mask].mean().item() if same_depth_mask.any() else 0
            diff_depth_attn = head_attn[diff_depth_mask].mean().item() if diff_depth_mask.any() else 0
            depth_bias = same_depth_attn / max(diff_depth_attn, 1e-8)

            # Same for type
            same_type_mask = types_t.unsqueeze(0) == types_t.unsqueeze(1)
            diff_type_mask = ~same_type_mask

            same_type_attn = head_attn[same_type_mask].mean().item() if same_type_mask.any() else 0
            diff_type_attn = head_attn[diff_type_mask].mean().item() if diff_type_mask.any() else 0
            type_bias = same_type_attn / max(diff_type_attn, 1e-8)

            spec = ""
            if depth_bias > 2.0:
                spec = "depth-focused"
            elif type_bias > 2.0:
                spec = "type-focused"
            elif depth_bias > 1.5 or type_bias > 1.5:
                spec = "mild bias"

            print(f"  L{layer_idx:<6} H{head_idx:<4} {depth_bias:>16.2f}x {type_bias:>16.2f}x {spec:>16}")

    # Summary
    all_depth_biases = []
    all_type_biases = []
    for attn in all_attn:
        for h in range(attn.shape[1]):
            ha = attn[0, h]
            same_d = depths_t.unsqueeze(0) == depths_t.unsqueeze(1)
            diff_d = ~same_d
            db = ha[same_d].mean().item() / max(ha[diff_d].mean().item(), 1e-8)
            all_depth_biases.append(db)

            same_t = types_t.unsqueeze(0) == types_t.unsqueeze(1)
            diff_t = ~same_t
            tb = ha[same_t].mean().item() / max(ha[diff_t].mean().item(), 1e-8)
            all_type_biases.append(tb)

    max_depth_bias = max(all_depth_biases)
    max_type_bias = max(all_type_biases)
    depth_specialized = sum(1 for b in all_depth_biases if b > 2.0)
    type_specialized = sum(1 for b in all_type_biases if b > 2.0)
    total_heads = len(all_depth_biases)

    print(f"\n  Summary:")
    print(f"    Max depth bias: {max_depth_bias:.2f}x (head attends {max_depth_bias:.1f}x more to same-depth nodes)")
    print(f"    Max type bias:  {max_type_bias:.2f}x")
    print(f"    Depth-specialized heads (>2x bias): {depth_specialized}/{total_heads}")
    print(f"    Type-specialized heads (>2x bias):  {type_specialized}/{total_heads}")

    if depth_specialized > 0 or type_specialized > 0:
        print(f"\n  PASS: Some heads show structural specialization")
    else:
        print(f"\n  FAIL: No heads show clear structural specialization (>2x bias)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_v5/vocab.json")
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

    print(f"Tree Positional Encoding — Claims Verification")
    print(f"Checkpoint: epoch {checkpoint['epoch']}")
    print()

    depth_emb = model.embedding.depth_embedding.weight.data
    sibling_emb = model.embedding.sibling_embedding.weight.data
    node_type_emb = model.embedding.node_type_embedding.weight.data

    test_uniqueness(depth_emb, sibling_emb, node_type_emb, config.max_depth, config.max_sibling)
    test_distance_sensitivity(depth_emb, sibling_emb, node_type_emb)
    test_decomposability(model, vocab)


if __name__ == "__main__":
    main()
