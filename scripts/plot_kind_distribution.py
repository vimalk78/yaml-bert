"""Plot the distribution of K8s kinds in the training corpus.

Shows two views side by side:
  - docs by kind (how many manifests per kind)
  - tokens by kind (how much of the training signal each kind contributed)

The contrast is the headline: CRDs are 3% of documents but 46% of training
tokens because each CRD is ~15× larger than a typical manifest. This biases
the model's representations toward schema-definition patterns.

Usage:
    python scripts/plot_kind_distribution.py output_v4/doc_cache.pkl \\
        --top 20 --out /tmp/kind_distribution.html
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import pickle
from collections import Counter

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from yaml_bert.types import NodeType


def _extract_kind(nodes) -> str:
    for i, n in enumerate(nodes):
        if (n.token == "kind" and n.depth == 0
                and n.node_type == NodeType.KEY and i + 1 < len(nodes)):
            return nodes[i + 1].token
    return "(no kind)"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training-corpus kind distribution")
    parser.add_argument("cache_pickle", type=str,
                        help="Path to doc_cache.pkl (e.g., output_v4/doc_cache.pkl)")
    parser.add_argument("--top", type=int, default=20, help="Show top N kinds")
    parser.add_argument("--highlight", nargs="+", default=["CustomResourceDefinition"],
                        help="Kinds to color differently in the bars")
    parser.add_argument("--out", type=str, default="docs/figures/kind_distribution.html")
    args = parser.parse_args()

    print(f"Loading {args.cache_pickle} ...")
    with open(args.cache_pickle, "rb") as f:
        docs = pickle.load(f)
    print(f"Loaded {len(docs):,} documents")

    doc_counts: Counter = Counter()
    tok_counts: Counter = Counter()
    for d in docs:
        k = _extract_kind(d)
        doc_counts[k] += 1
        tok_counts[k] += len(d)

    total_docs = sum(doc_counts.values())
    total_toks = sum(tok_counts.values())
    print(f"Total kinds: {len(doc_counts)}  |  total tokens: {total_toks:,}")

    # Pick the top N by document count for consistency between panels
    top_by_docs = [k for k, _ in doc_counts.most_common(args.top)]

    highlight_set = set(args.highlight)

    def bar_color(k: str, base: str) -> str:
        return "#d62728" if k in highlight_set else base

    docs_pct = [doc_counts[k] / total_docs * 100 for k in top_by_docs]
    toks_pct = [tok_counts[k] / total_toks * 100 for k in top_by_docs]
    avg_size = [tok_counts[k] / doc_counts[k] for k in top_by_docs]

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        horizontal_spacing=0.15,
        subplot_titles=("Share of documents", "Share of training tokens"),
    )

    fig.add_trace(go.Bar(
        x=docs_pct, y=top_by_docs, orientation="h",
        marker=dict(color=[bar_color(k, "#4c78a8") for k in top_by_docs],
                    line=dict(color="black", width=0.5)),
        text=[f"{p:.1f}% ({doc_counts[k]:,} docs)" for p, k in zip(docs_pct, top_by_docs)],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{customdata[0]:,} docs (%{x:.2f}% of corpus)<br>avg size: %{customdata[1]:.0f} tokens<extra></extra>",
        customdata=[(doc_counts[k], avg_size[i]) for i, k in enumerate(top_by_docs)],
        showlegend=False,
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=toks_pct, y=top_by_docs, orientation="h",
        marker=dict(color=[bar_color(k, "#f58518") for k in top_by_docs],
                    line=dict(color="black", width=0.5)),
        text=[f"{p:.1f}% ({tok_counts[k]:,})" for p, k in zip(toks_pct, top_by_docs)],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{y}</b><br>%{customdata[0]:,} tokens (%{x:.2f}% of training signal)<br>avg per doc: %{customdata[1]:.0f}<extra></extra>",
        customdata=[(tok_counts[k], avg_size[i]) for i, k in enumerate(top_by_docs)],
        showlegend=False,
    ), row=1, col=2)

    # Reverse Y so largest bars appear at top
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="% of documents", row=1, col=1, ticksuffix="%")
    fig.update_xaxes(title_text="% of training tokens", row=1, col=2, ticksuffix="%")

    fig.update_layout(
        title=(f"Training corpus by kind — {total_docs:,} docs, {total_toks:,} tokens, "
               f"{len(doc_counts)} distinct kinds (highlighted in red: {', '.join(args.highlight)})"),
        width=1800, height=900,
        template="plotly_white",
    )

    fig.write_html(args.out, include_plotlyjs="inline", config={"responsive": True})
    print(f"Saved {args.out}")

    # Print summary table
    print()
    print(f"{'Kind':<40s} {'docs%':>7} {'avg_size':>10} {'tokens%':>9}")
    print("-" * 70)
    for k in top_by_docs:
        d_pct = doc_counts[k] / total_docs * 100
        t_pct = tok_counts[k] / total_toks * 100
        avg = tok_counts[k] / doc_counts[k]
        marker = " *" if k in highlight_set else ""
        print(f"{k:<40s} {d_pct:6.1f}% {avg:9.0f} {t_pct:8.1f}%{marker}")


if __name__ == "__main__":
    main()
