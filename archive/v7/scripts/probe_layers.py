"""Probe what information each transformer layer encodes.

Trains linear classifiers on frozen hidden states at each layer to predict:
- depth (0-15): Does the model know how deep a node is?
- parent_key: Does the model know the parent key?
- kind: Does the model know the document kind?
- node_type: Does the model know if it's a key or value?

For v4: kind and parent are NOT in the input embedding.
Any probe accuracy for these was genuinely learned, not leaked.

Usage:
    PYTHONPATH=. python scripts/probe_layers.py output_v4/checkpoints/yaml_bert_v4_epoch_15.pt --vocab output_v4/vocab.json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from yaml_bert.config import YamlBertConfig
from yaml_bert.dataset import YamlDataset, _extract_kind
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary
from yaml_bert.types import NodeType


def extract_layer_states(
    model: YamlBertModel,
    dataset: YamlDataset,
    vocab: Vocabulary,
    max_docs: int = 300,
) -> dict:
    """Run forward pass, extract hidden states at embedding + each layer."""
    model.eval()
    type_map = {NodeType.KEY: 0, NodeType.VALUE: 1, NodeType.LIST_KEY: 2, NodeType.LIST_VALUE: 3}
    num_layers = len(model.encoder.layers)

    all_embedding: list[torch.Tensor] = []
    all_layers: dict[int, list[torch.Tensor]] = {l: [] for l in range(num_layers)}
    all_depths: list[int] = []
    all_node_types: list[int] = []
    all_parent_keys: list[int] = []
    all_kinds: list[int] = []

    for idx in range(min(max_docs, len(dataset))):
        item = dataset[idx]
        nodes = dataset.documents[idx]
        kind = _extract_kind(nodes)
        kind_id = vocab.encode_kind(kind)

        seq_len = item["token_ids"].shape[0]
        nodes = nodes[:seq_len]

        token_ids = item["token_ids"].unsqueeze(0)
        node_types_t = item["node_types"].unsqueeze(0)
        depths_t = item["depths"].unsqueeze(0)
        siblings_t = item["sibling_indices"].unsqueeze(0)

        with torch.no_grad():
            x = model.embedding(token_ids, node_types_t, depths_t, siblings_t)
            all_embedding.append(x.squeeze(0))

            for l, layer in enumerate(model.encoder.layers):
                x = layer(x)
                all_layers[l].append(x.squeeze(0))

        for node in nodes:
            all_depths.append(min(node.depth, 15))
            all_node_types.append(type_map[node.node_type])
            parent_key = Vocabulary.extract_parent_key(node.parent_path)
            all_parent_keys.append(vocab.encode_key(parent_key))
            all_kinds.append(kind_id)

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{min(max_docs, len(dataset))} docs")

    labels = {
        "depth": torch.tensor(all_depths),
        "node_type": torch.tensor(all_node_types),
        "parent_key": torch.tensor(all_parent_keys),
        "kind": torch.tensor(all_kinds),
    }

    layer_tensors = {l: torch.cat(v) for l, v in all_layers.items()}
    embedding_tensor = torch.cat(all_embedding)

    print(f"Extracted {len(all_depths)} nodes from {min(max_docs, len(dataset))} docs")
    return {
        "embedding": embedding_tensor,
        "layers": layer_tensors,
        "labels": labels,
        "num_layers": num_layers,
    }


def train_probe(states: torch.Tensor, labels: torch.Tensor, num_classes: int, epochs: int = 20) -> float:
    """Train a linear probe, return validation accuracy."""
    n = states.shape[0]
    perm = torch.randperm(n)
    split = int(n * 0.8)
    train_idx, val_idx = perm[:split], perm[split:]

    X_train, y_train = states[train_idx], labels[train_idx]
    X_val, y_val = states[val_idx], labels[val_idx]

    probe = nn.Linear(states.shape[1], num_classes)
    optimizer = torch.optim.Adam(probe.parameters(), lr=1e-3)

    for _ in range(epochs):
        probe.train()
        logits = probe(X_train)
        loss = F.cross_entropy(logits, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    probe.eval()
    with torch.no_grad():
        preds = probe(X_val).argmax(dim=-1)
        return (preds == y_val).float().mean().item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe layer representations")
    parser.add_argument("checkpoint", type=str)
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--yaml-dir", type=str, default="cluster-yamls",
                        help="Directory of YAMLs to probe on (recursive)")
    parser.add_argument("--max-docs", type=int, default=300)
    parser.add_argument("--cached-docs", type=str, default=None,
                        help="Path to cached docs pickle (faster than parsing)")
    parser.add_argument("--out", type=str, default="docs/figures/probe_layers.html",
                        help="HTML output path for the per-layer accuracy plot")
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
    print(f"Loaded epoch {checkpoint['epoch']}")

    # Load dataset
    if args.cached_docs:
        dataset = YamlDataset.from_cached_docs_v4(args.cached_docs, vocab)
    else:
        from yaml_bert.linearizer import YamlLinearizer
        from yaml_bert.annotator import DomainAnnotator
        dataset = YamlDataset(
            yaml_dir=args.yaml_dir, vocab=vocab,
            linearizer=YamlLinearizer(), annotator=DomainAnnotator(),
        )
    print(f"Dataset: {len(dataset)} docs from {args.yaml_dir}\n")

    print("Extracting hidden states...")
    data = extract_layer_states(model, dataset, vocab, max_docs=args.max_docs)

    num_layers = data["num_layers"]
    labels = data["labels"]

    num_classes = {
        "depth": 16,
        "node_type": 4,
        "parent_key": vocab.key_vocab_size,
        "kind": vocab.kind_vocab_size,
    }

    # Probe at each layer
    print(f"\n{'Layer':<12} {'Depth':>8} {'Type':>8} {'Parent':>8} {'Kind':>8}")
    print("-" * 48)

    properties = ["depth", "node_type", "parent_key", "kind"]
    # accuracies[prop] = [embedding_acc, layer0_acc, layer1_acc, ...]
    accuracies: dict[str, list[float]] = {p: [] for p in properties}
    stage_names: list[str] = []

    # Embedding layer
    accs = []
    for prop in properties:
        acc = train_probe(data["embedding"], labels[prop], num_classes[prop])
        accs.append(acc)
        accuracies[prop].append(acc)
    stage_names.append("Embedding")
    print(f"{'Embedding':<12} {accs[0]:>7.1%} {accs[1]:>7.1%} {accs[2]:>7.1%} {accs[3]:>7.1%}")

    # Transformer layers
    for l in range(num_layers):
        accs = []
        for prop in properties:
            acc = train_probe(data["layers"][l], labels[prop], num_classes[prop])
            accs.append(acc)
            accuracies[prop].append(acc)
        stage_names.append(f"Layer {l}")
        print(f"{'Layer ' + str(l):<12} {accs[0]:>7.1%} {accs[1]:>7.1%} {accs[2]:>7.1%} {accs[3]:>7.1%}")

    print(f"\nNote: kind and parent_key are NOT in the v4 input embedding.")
    print(f"Any accuracy above random for these was learned from the prediction target.")
    print(f"Random baselines: depth={1/16:.1%}, type={1/4:.1%}, parent=~{1/100:.1%}, kind=~{1/vocab.kind_vocab_size:.1%}")

    # Plot per-layer accuracy as interactive HTML
    _plot_probe_accuracies(stage_names, accuracies, num_classes, args.out)


def _plot_probe_accuracies(stage_names: list[str], accuracies: dict[str, list[float]],
                           num_classes: dict[str, int], out: str) -> None:
    """Emit an interactive HTML heatmap showing probe accuracy across layers."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # Row order chosen so the "always recoverable" features sit at top and
    # the "learned through attention" features sit at the bottom — eye reads
    # the visible gradient on the kind row.
    props = ["node_type", "depth", "parent_key", "kind"]
    pretty_label = {
        "depth": f"Depth<br><span style='font-size:0.75em;color:#888'>{num_classes['depth']} classes · rand {1/num_classes['depth']:.0%}</span>",
        "node_type": f"Node type<br><span style='font-size:0.75em;color:#888'>{num_classes['node_type']} classes · rand {1/num_classes['node_type']:.0%}</span>",
        "parent_key": f"Parent key<br><span style='font-size:0.75em;color:#888'>{num_classes['parent_key']} classes · rand {1/num_classes['parent_key']:.0%}</span>",
        "kind": f"Kind<br><span style='font-size:0.75em;color:#888'>{num_classes['kind']} classes · rand {1/num_classes['kind']:.0%}</span>",
    }
    role_label = {
        "depth": "in input",
        "node_type": "in input",
        "parent_key": "learned",
        "kind": "learned",
    }

    z = [[accuracies[p][s] * 100 for s in range(len(stage_names))] for p in props]
    text = [[f"{accuracies[p][s] * 100:.1f}%" for s in range(len(stage_names))] for p in props]
    rand_baselines = [f"{(1 / num_classes[p]) * 100:.1f}%" for p in props]

    deltas = [accuracies[p][-1] * 100 - accuracies[p][0] * 100 for p in props]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.88, 0.12],
        horizontal_spacing=0.02,
        subplot_titles=("Probe accuracy by stage", "Δ (final − embedding)"),
        specs=[[{"type": "heatmap"}, {"type": "bar"}]],
    )

    # Main heatmap
    fig.add_trace(go.Heatmap(
        z=z,
        x=stage_names,
        y=[pretty_label[p] for p in props],
        text=text,
        texttemplate="%{text}",
        textfont=dict(size=14, color="black"),
        colorscale=[
            [0.0, "#fff7bc"],   # pale yellow (chance)
            [0.5, "#fec44f"],   # amber (middling)
            [0.85, "#d95f0e"],  # orange (high)
            [1.0, "#7f2704"],   # dark brown (near-perfect)
        ],
        zmin=0, zmax=100,
        colorbar=dict(title="Accuracy %", x=0.85, len=0.85),
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.1f}%<br>random baseline shown in row label<extra></extra>",
    ), row=1, col=1)

    # Delta bar chart on the right
    bar_colors = ["#5b8def" if d >= 0 else "#d62728" for d in deltas]
    fig.add_trace(go.Bar(
        y=[pretty_label[p] for p in props],
        x=deltas,
        orientation="h",
        marker=dict(color=bar_colors, line=dict(color="black", width=1)),
        text=[f"{d:+.1f}%" for d in deltas],
        textposition="outside",
        textfont=dict(size=13),
        showlegend=False,
        hovertemplate="<b>%{y}</b><br>Δ: %{x:+.1f}%<extra></extra>",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Encoder stage", row=1, col=1)
    fig.update_xaxes(title_text="Δ (pp)", range=[min(deltas) - 8, max(deltas) + 12], row=1, col=2)
    fig.update_yaxes(showticklabels=False, row=1, col=2)

    fig.update_layout(
        title=("What does each encoder stage encode? "
               "Linear probe accuracy across the 4 structural properties"),
        width=1800, height=620,
        template="plotly_white",
    )
    fig.write_html(out, include_plotlyjs="inline", config={"responsive": True})
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
