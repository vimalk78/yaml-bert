"""Interactive HTML plots of YAML-BERT embeddings.

Each --view answers one question about what the model has learned. All views
share the same forward-pass extraction; just the slicing and coloring differ.
Outputs self-contained HTML — open in a browser and hover for per-point
metadata (token, kind, depth, parent_path, adjacent value).

Views:
  token-identity     Static vs contextualized for a handful of tokens.
                     The "soul vs body" story: input embeddings look random,
                     contextualized vectors form tight per-token clusters.

  polysemous         One ambiguous token (default: 'name') colored by kind.
                     Tests whether the model preserves role-specific meaning:
                     metadata.name vs containers[i].name vs ports[i].name
                     should fracture into sub-clusters.

  single-kind        Within one kind (default: Deployment), all key positions
                     colored by token. Shows internal structural organization.
                     Hover shows parent_path — explains the sub-clusters.

For "how does meaning evolve across layers" specifically, use
scripts/probe_layers.py — linear probes at each layer give a far cleaner
evolution story than t-SNE could.

Usage:
  python scripts/plot_contextualized_embeddings.py <checkpoint> \\
      --vocab <vocab.json> --view single-kind --yaml-dir cluster-yamls \\
      --sample 500 --out /tmp/deployment.html
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

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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
    parent_path: str
    next_value: str        # adjacent value, if any
    source: str            # YAML file path
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


def extract_positions(model, vocab, yaml_files: Sequence[str],
                      layers: Sequence[int]) -> list[Position]:
    """Run each YAML through the model and capture hidden states at chosen layers."""
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
            next_val = ""
            if i + 1 < len(nodes) and nodes[i + 1].node_type in (NodeType.VALUE, NodeType.LIST_VALUE):
                next_val = nodes[i + 1].token[:60]
            pos = Position(
                token=n.token,
                kind=kind,
                depth=min(n.depth, 15),
                sibling=min(n.sibling_index, 31),
                parent_path=n.parent_path or "(root)",
                next_value=next_val,
                source=os.path.basename(path),
                hiddens={li: captured[li][0, i].numpy() for li in layers},
            )
            positions.append(pos)

    for h in handles:
        h.remove()

    print(f"Extracted {len(positions)} key positions from "
          f"{len(yaml_files) - skipped}/{len(yaml_files)} files ({skipped} skipped).")
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


def _hover_template() -> str:
    return (
        "<b>%{customdata[0]}</b> "
        "(kind: %{customdata[1]}, depth: %{customdata[2]})<br>"
        "path: %{customdata[3]}<br>"
        "value: %{customdata[4]}<br>"
        "<i>%{customdata[5]}</i>"
        "<extra></extra>"
    )


def _customdata(positions: list[Position]) -> np.ndarray:
    return np.array([
        [p.token, p.kind, p.depth, p.parent_path, p.next_value, p.source]
        for p in positions
    ], dtype=object)


# -------- Views --------

def _write_html(fig, out: str) -> None:
    fig.write_html(out, include_plotlyjs="inline", config={"responsive": True})


def view_token_identity(positions: list[Position], emb, vocab, tokens, out: str, layer: int,
                        width: int = 1800, height: int = 1100) -> None:
    picks = [t for t in tokens if sum(1 for p in positions if p.token == t) >= 3]
    if not picks:
        print("No tokens with ≥3 occurrences in this corpus.")
        return
    print(f"Plotting tokens: {picks}")

    ctx_positions = [p for p in positions if p.token in set(picks)]
    ctx_X = np.array([p.hiddens[layer] for p in ctx_positions])
    ctx_labels = [p.token for p in ctx_positions]
    w, b = _within_between_cosine(ctx_X, ctx_labels)
    print(f"CTX cosine — within-token: {w:.3f}, between-token: {b:.3f}")

    ctx_2d = _tsne_2d(ctx_X, perplexity=15)

    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(picks)}

    fig = go.Figure()
    for tok in picks:
        mask = [i for i, p in enumerate(ctx_positions) if p.token == tok]
        sub = [ctx_positions[i] for i in mask]
        fig.add_trace(go.Scatter(
            x=ctx_2d[mask, 0], y=ctx_2d[mask, 1],
            mode="markers",
            name=f"{tok} (n={len(sub)})",
            marker=dict(size=8, color=color_map[tok], line=dict(width=0.5, color="black")),
            customdata=_customdata(sub),
            hovertemplate=_hover_template(),
        ))

    fig.update_layout(
        title=(f"Contextualized hidden states by token (layer {layer}) — "
               f"within-token {w:.2f} vs between-token {b:.2f}"),
        width=width, height=height,
        hovermode="closest",
        template="plotly_white",
    )
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False)
    _write_html(fig, out)
    print(f"Saved {out}")


def view_polysemous(positions, token: str, out: str, layer: int,
                    top_kinds: int = 8, kinds_filter: list[str] | None = None,
                    width: int = 1800, height: int = 1100) -> None:
    matches = [p for p in positions if p.token == token and p.kind != "?"]
    if not matches:
        print(f"No occurrences of '{token}' in this corpus.")
        return

    counts: dict[str, int] = defaultdict(int)
    for p in matches:
        counts[p.kind] += 1
    if kinds_filter:
        kinds = [k for k in kinds_filter if k in counts]
    else:
        kinds = sorted(counts.keys(), key=lambda k: -counts[k])[:top_kinds]
    print(f"'{token}' in top kinds: {[(k, counts[k]) for k in kinds]}")

    matches = [p for p in matches if p.kind in set(kinds)]
    X = np.array([p.hiddens[layer] for p in matches])
    kind_labels = [p.kind for p in matches]
    if len(X) < 5:
        print("Too few points to plot.")
        return

    w, b = _within_between_cosine(X, kind_labels)
    print(f"CTX cosine (by kind) — within {w:.3f} vs between {b:.3f}")

    X_2d = _tsne_2d(X, perplexity=20)
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    color_map = {k: palette[i % len(palette)] for i, k in enumerate(kinds)}

    fig = go.Figure()
    for k in kinds:
        idx = [i for i, p in enumerate(matches) if p.kind == k]
        if not idx:
            continue
        sub = [matches[i] for i in idx]
        fig.add_trace(go.Scatter(
            x=X_2d[idx, 0], y=X_2d[idx, 1],
            mode="markers",
            name=f"{k} (n={len(sub)})",
            marker=dict(size=8, color=color_map[k], line=dict(width=0.5, color="black")),
            customdata=_customdata(sub),
            hovertemplate=_hover_template(),
        ))

    fig.update_layout(
        title=f"'{token}' across kinds (layer {layer}) — within-kind {w:.2f} vs between-kind {b:.2f}",
        width=width, height=height,
        hovermode="closest",
        template="plotly_white",
    )
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False)
    _write_html(fig, out)
    print(f"Saved {out}")


def view_single_kind(positions, kind: str, tokens, out: str, layer: int,
                     width: int = 1800, height: int = 1100) -> None:
    in_kind = [p for p in positions if p.kind == kind]
    if not in_kind:
        print(f"No positions found for kind={kind}.")
        return
    picks = [t for t in tokens if sum(1 for p in in_kind if p.token == t) >= 3]
    if not picks:
        print(f"No requested tokens with ≥3 occurrences in {kind}.")
        return
    print(f"Plotting {picks} within {kind} ({len(in_kind)} positions total)")

    sub = [p for p in in_kind if p.token in set(picks)]
    X = np.array([p.hiddens[layer] for p in sub])
    labels = [p.token for p in sub]
    w, b = _within_between_cosine(X, labels)
    print(f"CTX cosine — within-token {w:.3f} vs between-token {b:.3f}")

    X_2d = _tsne_2d(X, perplexity=15)
    palette = px.colors.qualitative.Plotly + px.colors.qualitative.D3
    color_map = {t: palette[i % len(palette)] for i, t in enumerate(picks)}

    fig = go.Figure()
    for tok in picks:
        idx = [i for i, p in enumerate(sub) if p.token == tok]
        if not idx:
            continue
        sub_tok = [sub[i] for i in idx]
        fig.add_trace(go.Scatter(
            x=X_2d[idx, 0], y=X_2d[idx, 1],
            mode="markers",
            name=f"{tok} (n={len(sub_tok)})",
            marker=dict(size=9, color=color_map[tok], line=dict(width=0.5, color="black")),
            customdata=_customdata(sub_tok),
            hovertemplate=_hover_template(),
        ))

    fig.update_layout(
        title=f"Internal organization within '{kind}' (layer {layer}) — within {w:.2f} vs between {b:.2f}",
        width=width, height=height,
        hovermode="closest",
        template="plotly_white",
    )
    fig.update_xaxes(showticklabels=False, zeroline=False)
    fig.update_yaxes(showticklabels=False, zeroline=False)
    _write_html(fig, out)
    print(f"Saved {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot YAML-BERT embeddings interactively")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument(
        "--view",
        choices=["token-identity", "polysemous", "single-kind"],
        default="token-identity",
    )
    parser.add_argument("--yaml-dir", type=str, default="cluster-yamls")
    parser.add_argument("--sample", type=int, default=500, help="Random sample of N YAMLs (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer", type=int, default=-1)
    parser.add_argument("--tokens", nargs="+", default=DEFAULT_TOKENS)
    parser.add_argument("--token", type=str, default="name")
    parser.add_argument("--kind", type=str, default="Deployment")
    parser.add_argument("--top-kinds", type=int, default=8)
    parser.add_argument("--kinds-filter", nargs="+", default=None)
    parser.add_argument("--width", type=int, default=None, help="Plot width in px (defaults vary by view)")
    parser.add_argument("--height", type=int, default=None, help="Plot height in px (defaults vary by view)")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    out = args.out or f"/tmp/yaml_bert_{args.view}.html"

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

    layer_idx = args.layer if args.layer >= 0 else n_encoder_layers + args.layer
    layers = [layer_idx]

    positions = extract_positions(model, vocab, yaml_files, layers=layers)

    size_kwargs = {}
    if args.width is not None:
        size_kwargs["width"] = args.width
    if args.height is not None:
        size_kwargs["height"] = args.height

    if args.view == "token-identity":
        view_token_identity(positions, emb, vocab, args.tokens, out, layer=layers[0], **size_kwargs)
    elif args.view == "polysemous":
        view_polysemous(positions, args.token, out, layer=layers[0],
                        top_kinds=args.top_kinds, kinds_filter=args.kinds_filter,
                        **size_kwargs)
    elif args.view == "single-kind":
        view_single_kind(positions, args.kind, args.tokens, out, layer=layers[0], **size_kwargs)


if __name__ == "__main__":
    main()
