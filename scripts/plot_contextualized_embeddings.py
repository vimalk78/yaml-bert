"""Plot YAML-BERT embeddings from multiple angles.

Each --view answers one question about what the model has learned. All views
share the same forward-pass extraction; just the slicing and coloring differ.

Views:
  token-identity     Static vs contextualized for a handful of tokens.
                     The "soul vs body" story: input embeddings look random,
                     contextualized vectors form tight per-token clusters.

  polysemous         One ambiguous token (default: 'name') colored by kind.
                     Tests whether the model preserves role-specific meaning:
                     metadata.name vs containers[i].name vs ports[i].name
                     should fracture into sub-clusters.

  single-kind        Within one kind (default: Deployment), all key positions
                     colored by token. Shows internal structural organization
                     of one resource type.

  layer-evolution    One token's hidden state across layers (1, 3, 5) plus
                     its static input embedding. Visualizes how meaning
                     emerges through the encoder.

Usage:
  python scripts/plot_contextualized_embeddings.py <checkpoint> \\
      --vocab <vocab.json> --view token-identity --yaml-dir cluster-yamls \\
      --sample 500
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import glob
import os
import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.types import NodeType
from yaml_bert.vocab import Vocabulary

_TYPE_MAP = {
    NodeType.KEY: 0,
    NodeType.VALUE: 1,
    NodeType.LIST_KEY: 2,
    NodeType.LIST_VALUE: 3,
}

DEFAULT_TOKENS = [
    "name", "replicas", "containers", "image", "metadata",
    "spec", "selector", "ports", "labels", "namespace",
]


@dataclass
class Position:
    token: str
    kind: str
    depth: int
    sibling: int
    hiddens: dict = field(default_factory=dict)  # layer_idx -> np.ndarray


def _load_model(checkpoint_path: str, vocab_path: str):
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
    return model, vocab, emb


def _detect_kind(nodes) -> str:
    for i, n in enumerate(nodes):
        if (n.token == "kind" and n.depth == 0
                and n.node_type == NodeType.KEY and i + 1 < len(nodes)):
            return nodes[i + 1].token
    return "?"


def extract_positions(model, vocab, yaml_files: Sequence[str], layers: Sequence[int]) -> list[Position]:
    """Run each YAML through the model, capture hidden states at chosen layers.

    Only KEY/LIST_KEY positions are captured (matches what the model predicts).
    """
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(_mod, _inp, out):
            captured[layer_idx] = out.detach()
        return hook

    handles = []
    for li in layers:
        h = model.encoder.layers[li].register_forward_hook(make_hook(li))
        handles.append(h)

    positions: list[Position] = []
    skipped = 0
    for path in yaml_files:
        try:
            with open(path) as f:
                text = f.read()
            nodes = linearizer.linearize(text)
        except Exception:
            skipped += 1
            continue
        if not nodes:
            skipped += 1
            continue
        annotator.annotate(nodes)
        kind = _detect_kind(nodes)

        token_ids, node_types, depths, sibs = [], [], [], []
        for n in nodes:
            if n.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                token_ids.append(vocab.encode_key(n.token))
            else:
                token_ids.append(vocab.encode_value(n.token))
            node_types.append(_TYPE_MAP[n.node_type])
            depths.append(min(n.depth, 15))
            sibs.append(min(n.sibling_index, 31))

        t = lambda x: torch.tensor([x])
        with torch.no_grad():
            model(t(token_ids), t(node_types), t(depths), t(sibs))

        for i, n in enumerate(nodes):
            if n.node_type not in (NodeType.KEY, NodeType.LIST_KEY):
                continue
            pos = Position(
                token=n.token,
                kind=kind,
                depth=min(n.depth, 15),
                sibling=min(n.sibling_index, 31),
                hiddens={li: captured[li][0, i].numpy() for li in layers},
            )
            positions.append(pos)

    for h in handles:
        h.remove()

    print(f"Extracted {len(positions)} key positions from {len(yaml_files) - skipped}/{len(yaml_files)} files ({skipped} skipped).")
    return positions


def _tsne_2d(X: np.ndarray, perplexity: int = 30) -> np.ndarray:
    return TSNE(
        n_components=2,
        random_state=42,
        perplexity=min(perplexity, max(2, len(X) - 1)),
        max_iter=1000,
        init="pca" if X.shape[1] >= 2 else "random",
    ).fit_transform(X)


def _within_between_cosine(X: np.ndarray, labels: Sequence[str]) -> tuple[float, float]:
    sim = cosine_similarity(X)
    within, between = [], []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            (within if labels[i] == labels[j] else between).append(sim[i, j])
    return float(np.mean(within)) if within else 0.0, float(np.mean(between)) if between else 0.0


# -------- Views --------

def view_token_identity(positions: list[Position], emb, vocab, tokens, out: str, layer: int) -> None:
    picks = [t for t in tokens if sum(1 for p in positions if p.token == t) >= 3]
    if not picks:
        print("No tokens with ≥3 occurrences in this corpus.")
        return
    print(f"Plotting tokens: {picks}")

    ctx_X, ctx_labels = [], []
    static_X = []
    for tok in picks:
        for p in positions:
            if p.token == tok:
                ctx_X.append(p.hiddens[layer])
                ctx_labels.append(tok)
        static_X.append(emb.key_embedding.weight[vocab.encode_key(tok)].detach().numpy())

    ctx_X = np.array(ctx_X)
    static_X = np.array(static_X)

    w, b = _within_between_cosine(ctx_X, ctx_labels)
    print(f"CTX cosine — within-token mean: {w:.3f}, between-token mean: {b:.3f}")

    ctx_2d = _tsne_2d(ctx_X, perplexity=15)
    static_2d = _tsne_2d(static_X, perplexity=min(5, len(picks) - 1))

    colors = plt.cm.tab10(np.linspace(0, 1, len(picks)))
    color_map = {t: c for t, c in zip(picks, colors)}

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    for tok in picks:
        idx = [i for i, lab in enumerate(ctx_labels) if lab == tok]
        axes[0].scatter(ctx_2d[idx, 0], ctx_2d[idx, 1],
                        c=[color_map[tok]], s=50, alpha=0.7,
                        label=f"{tok} (n={len(idx)})",
                        edgecolors="black", linewidths=0.4)
    axes[0].set_title(f"Contextualized (layer {layer}) — within {w:.2f} vs between {b:.2f}")
    axes[0].legend(loc="best", fontsize=8)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    for i, tok in enumerate(picks):
        axes[1].scatter(static_2d[i, 0], static_2d[i, 1],
                        c=[color_map[tok]], s=120, alpha=0.9,
                        edgecolors="black", linewidths=0.8)
        axes[1].annotate(tok, (static_2d[i, 0], static_2d[i, 1]),
                         fontsize=8, xytext=(5, 5), textcoords="offset points")
    axes[1].set_title("Static (input embedding table)")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.suptitle("Where meaning lives: static vs contextualized")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def view_polysemous(positions: list[Position], token: str, out: str, layer: int) -> None:
    matches = [p for p in positions if p.token == token]
    if not matches:
        print(f"No occurrences of '{token}' in this corpus.")
        return
    kinds = sorted({p.kind for p in matches})
    kinds = [k for k in kinds if k != "?"]
    print(f"'{token}' found in kinds: {kinds}")

    X = np.array([p.hiddens[layer] for p in matches if p.kind != "?"])
    kind_labels = [p.kind for p in matches if p.kind != "?"]
    if len(X) < 5:
        print("Too few points to plot meaningfully.")
        return

    w, b = _within_between_cosine(X, kind_labels)
    print(f"CTX cosine (grouping by kind) — within {w:.3f} vs between {b:.3f}")

    X_2d = _tsne_2d(X, perplexity=20)

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(kinds), 1)))
    color_map = {k: c for k, c in zip(kinds, colors)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for k in kinds:
        idx = [i for i, kl in enumerate(kind_labels) if kl == k]
        if not idx:
            continue
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                   c=[color_map[k]], s=50, alpha=0.7,
                   label=f"{k} (n={len(idx)})",
                   edgecolors="black", linewidths=0.4)
    ax.set_title(f"'{token}' across kinds — within-kind {w:.2f} vs between-kind {b:.2f}")
    ax.legend(loc="best", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def view_single_kind(positions: list[Position], kind: str, tokens, out: str, layer: int) -> None:
    in_kind = [p for p in positions if p.kind == kind]
    if not in_kind:
        print(f"No positions found for kind={kind}.")
        return
    picks = [t for t in tokens if sum(1 for p in in_kind if p.token == t) >= 3]
    if not picks:
        print(f"No requested tokens met ≥3 occurrences in {kind}.")
        return
    print(f"Plotting {picks} within {kind} ({len(in_kind)} positions total)")

    X, labels = [], []
    for tok in picks:
        for p in in_kind:
            if p.token == tok:
                X.append(p.hiddens[layer])
                labels.append(tok)
    X = np.array(X)

    w, b = _within_between_cosine(X, labels)
    print(f"CTX cosine — within-token {w:.3f} vs between-token {b:.3f}")

    X_2d = _tsne_2d(X, perplexity=15)
    colors = plt.cm.tab10(np.linspace(0, 1, len(picks)))
    color_map = {t: c for t, c in zip(picks, colors)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for tok in picks:
        idx = [i for i, l in enumerate(labels) if l == tok]
        ax.scatter(X_2d[idx, 0], X_2d[idx, 1],
                   c=[color_map[tok]], s=50, alpha=0.7,
                   label=f"{tok} (n={len(idx)})",
                   edgecolors="black", linewidths=0.4)
    ax.set_title(f"Internal organization within '{kind}' — within {w:.2f} vs between {b:.2f}")
    ax.legend(loc="best", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def view_layer_evolution(positions: list[Position], emb, vocab, token: str, layers, out: str) -> None:
    matches = [p for p in positions if p.token == token]
    if len(matches) < 5:
        print(f"Need ≥5 occurrences of '{token}'; only {len(matches)} found.")
        return
    print(f"Plotting evolution of '{token}' across layers {layers} ({len(matches)} occurrences)")

    n_panels = 1 + len(layers)  # static + each layer
    fig, axes = plt.subplots(1, n_panels, figsize=(4 * n_panels, 5))

    # Color by kind so we can see context-specific drift across layers
    kinds = sorted({p.kind for p in matches if p.kind != "?"})
    colors = plt.cm.tab20(np.linspace(0, 1, max(len(kinds), 1)))
    color_map = {k: c for k, c in zip(kinds, colors)}

    # Panel 0: static (one point, but we duplicate to plot scale)
    static_vec = emb.key_embedding.weight[vocab.encode_key(token)].detach().numpy()
    axes[0].scatter([0], [0], c="gray", s=300, edgecolors="black")
    axes[0].annotate(token, (0, 0), ha="center", va="center", fontsize=11)
    axes[0].set_title("Static input")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    for ax_idx, li in enumerate(layers, start=1):
        X = np.array([p.hiddens[li] for p in matches if p.kind != "?"])
        kind_labels = [p.kind for p in matches if p.kind != "?"]
        if len(X) < 3:
            axes[ax_idx].set_title(f"Layer {li} (too few)")
            continue
        X_2d = _tsne_2d(X, perplexity=min(15, max(2, len(X) - 1)))
        for k in kinds:
            idx = [i for i, kl in enumerate(kind_labels) if kl == k]
            if not idx:
                continue
            axes[ax_idx].scatter(X_2d[idx, 0], X_2d[idx, 1],
                                 c=[color_map[k]], s=40, alpha=0.7,
                                 label=k if ax_idx == 1 else None,
                                 edgecolors="black", linewidths=0.3)
        w, b = _within_between_cosine(X, kind_labels)
        axes[ax_idx].set_title(f"Layer {li} (w/b cos: {w:.2f}/{b:.2f})")
        axes[ax_idx].set_xticks([]); axes[ax_idx].set_yticks([])

    if any(p.kind != "?" for p in matches):
        axes[1].legend(loc="upper left", fontsize=7, bbox_to_anchor=(0, -0.05), ncol=4)

    fig.suptitle(f"How '{token}' evolves through the encoder")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot YAML-BERT embeddings from multiple angles")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--view", choices=["token-identity", "polysemous", "single-kind", "layer-evolution"], default="token-identity")
    parser.add_argument("--yaml-dir", type=str, default="cluster-yamls")
    parser.add_argument("--sample", type=int, default=500, help="Random sample of N YAMLs (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--tokens", nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--token", type=str, default="name", help="Single token for polysemous / layer-evolution views")
    parser.add_argument("--kind", type=str, default="Deployment", help="Kind for single-kind view")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out = args.out or f"/tmp/yaml_bert_{args.view}.png"

    yaml_files = sorted(glob.glob(os.path.join(args.yaml_dir, "**", "*.yaml"), recursive=True))
    if not yaml_files:
        print(f"No YAMLs found under {args.yaml_dir}/")
        return
    print(f"Found {len(yaml_files)} YAMLs under {args.yaml_dir}/")

    random.seed(args.seed)
    if args.sample > 0 and len(yaml_files) > args.sample:
        yaml_files = random.sample(yaml_files, args.sample)
        print(f"Sampled {args.sample}")

    model, vocab, emb = _load_model(args.checkpoint, args.vocab)
    n_encoder_layers = len(model.encoder.layers)

    if args.view == "layer-evolution":
        layers = [0, n_encoder_layers // 2, n_encoder_layers - 1]
    else:
        layer_idx = args.layer if args.layer >= 0 else n_encoder_layers + args.layer
        layers = [layer_idx]

    positions = extract_positions(model, vocab, yaml_files, layers=layers)

    if args.view == "token-identity":
        view_token_identity(positions, emb, vocab, args.tokens, out, layer=layers[0])
    elif args.view == "polysemous":
        view_polysemous(positions, args.token, out, layer=layers[0])
    elif args.view == "single-kind":
        view_single_kind(positions, args.kind, args.tokens, out, layer=layers[0])
    elif args.view == "layer-evolution":
        view_layer_evolution(positions, emb, vocab, args.token, layers, out)


if __name__ == "__main__":
    main()
