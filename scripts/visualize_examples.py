"""Visualize attention patterns for interesting K8s resource types.

Usage:
    python visualize_examples.py output_hf/checkpoints/yaml_bert_epoch_10.pt
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
import sys

import yaml
from datasets import load_dataset

from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary, VocabBuilder
from yaml_bert.types import NodeType

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def find_examples(ds) -> dict[str, int]:
    """Find good example documents for each resource type."""
    targets: dict[str, int | None] = {
        "Endpoints": None,
        "Service": None,
        "StatefulSet": None,
        "NetworkPolicy": None,
    }
    volume_doc: int | None = None

    for i in range(len(ds)):
        try:
            doc = yaml.safe_load(ds[i]["content"])
        except Exception:
            continue
        if not isinstance(doc, dict):
            continue

        kind: str = doc.get("kind", "")
        content: str = ds[i]["content"]
        lines: int = content.count("\n")

        if kind in targets and targets[kind] is None and 15 < lines < 60:
            targets[kind] = i
            print(f"Found {kind} at idx={i} ({lines} lines)")

        if volume_doc is None and kind == "Pod" and "volumeMounts" in content and "volumes" in content and 20 < lines < 50:
            volume_doc = i
            print(f"Found Pod+volumes at idx={i} ({lines} lines)")

        if all(v is not None for v in targets.values()) and volume_doc is not None:
            break

    result: dict[str, int] = {}
    for kind, idx in targets.items():
        if idx is not None:
            result[kind] = idx
    if volume_doc is not None:
        result["Pod+volumes"] = volume_doc
    return result


def analyze_attention(
    model: YamlBertModel,
    vocab: Vocabulary,
    linearizer: YamlLinearizer,
    annotator: DomainAnnotator,
    yaml_text: str,
    label: str,
    output_dir: str,
    max_nodes: int = 35,
) -> None:
    """Analyze and visualize attention for a single YAML document."""
    nodes = linearizer.linearize(yaml_text)
    if not nodes:
        print(f"  Skipping {label}: no nodes")
        return
    annotator.annotate(nodes)

    if len(nodes) > max_nodes:
        nodes = nodes[:max_nodes]

    # Build labels
    token_labels: list[str] = []
    for n in nodes:
        prefix: str = "=" if n.node_type in (NodeType.VALUE, NodeType.LIST_VALUE) else ""
        token_labels.append(f"{prefix}{n.token[:25]}")

    # Encode
    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
    token_ids, node_types, depths, siblings, parent_keys = [], [], [], [], []
    for node in nodes:
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            token_ids.append(vocab.encode_key(node.token))
        else:
            token_ids.append(vocab.encode_value(node.token))
        node_types.append(type_map[node.node_type])
        depths.append(min(node.depth, 15))
        siblings.append(min(node.sibling_index, 31))
        parent_keys.append(vocab.encode_key(Vocabulary.extract_parent_key(node.parent_path)))

    t = lambda x: torch.tensor([x])
    attn = model.get_attention_weights(
        t(token_ids), t(node_types), t(depths), t(siblings), t(parent_keys)
    )

    # Find top off-diagonal attention patterns
    print(f"\n{'=' * 60}")
    print(f"{label}")
    print(f"{'=' * 60}")
    print(f"Nodes: {len(nodes)}")

    results: list[tuple[float, int, int, int, int]] = []
    for layer_idx, layer_w in enumerate(attn):
        w = layer_w[0]
        for head_idx in range(w.shape[0]):
            h = w[head_idx]
            off_diag = h.clone()
            off_diag.fill_diagonal_(0)
            max_val = off_diag.max().item()
            max_idx = off_diag.argmax().item()
            from_idx = max_idx // len(nodes)
            to_idx = max_idx % len(nodes)
            results.append((max_val, layer_idx, head_idx, from_idx, to_idx))

    results.sort(reverse=True)
    print(f"\nTop 5 attention patterns:")
    for val, li, hi, fi, ti in results[:5]:
        from_label = token_labels[fi]
        to_label = token_labels[ti]
        print(f"  L{li}H{hi}: {from_label} -> {to_label} (attn={val:.3f})")

    # Save plots for top 3 most interesting heads
    doc_dir: str = os.path.join(output_dir, label.replace("+", "_").replace(" ", "_").lower())
    os.makedirs(doc_dir, exist_ok=True)

    seen_heads: set[tuple[int, int]] = set()
    for val, li, hi, fi, ti in results[:3]:
        if (li, hi) in seen_heads:
            continue
        seen_heads.add((li, hi))

        weights = attn[li][0][hi].cpu()
        fig, ax = plt.subplots(figsize=(12, 10))
        im = ax.imshow(weights.numpy(), cmap="Blues", vmin=0)
        ax.set_title(f"{label} — Layer {li}, Head {hi}", fontsize=14)
        ax.set_xticks(range(len(token_labels)))
        ax.set_yticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(token_labels, fontsize=8)
        ax.set_xlabel("Attending to")
        ax.set_ylabel("Attending from")
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(os.path.join(doc_dir, f"L{li}H{hi}.png"), dpi=150)
        plt.close(fig)

    # Average attention across all heads and layers
    all_weights = torch.stack([layer_w[0].mean(dim=0) for layer_w in attn]).mean(dim=0).cpu()
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(all_weights.numpy(), cmap="Blues", vmin=0)
    ax.set_title(f"{label} — Average Attention (all layers, all heads)", fontsize=14)
    ax.set_xticks(range(len(token_labels)))
    ax.set_yticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(token_labels, fontsize=8)
    ax.set_xlabel("Attending to")
    ax.set_ylabel("Attending from")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(doc_dir, f"average.png"), dpi=150)
    plt.close(fig)

    print(f"Plots saved to: {doc_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, default="output_hf/vocab.json")
    parser.add_argument("--output-dir", type=str, default="output_hf/attention_examples")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading vocabulary and model...")
    vocab: Vocabulary = Vocabulary.load(args.vocab)
    config: YamlBertConfig = YamlBertConfig()
    emb = YamlBertEmbedding(config=config, key_vocab_size=vocab.key_vocab_size, value_vocab_size=vocab.value_vocab_size)
    model = YamlBertModel(config=config, embedding=emb, key_vocab_size=vocab.key_vocab_size)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded epoch {checkpoint['epoch']}")

    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    print("\nFinding example documents...")
    ds = load_dataset("substratusai/the-stack-yaml-k8s", split="train")
    examples: dict[str, int] = find_examples(ds)

    for label, idx in examples.items():
        yaml_text: str = ds[idx]["content"]
        print(f"\n--- {label} (idx={idx}) ---")
        print(yaml_text[:200] + "..." if len(yaml_text) > 200 else yaml_text)
        analyze_attention(
            model, vocab, linearizer, annotator,
            yaml_text, label, args.output_dir,
        )

    print(f"\nAll done! Results in: {args.output_dir}/")


if __name__ == "__main__":
    main()
