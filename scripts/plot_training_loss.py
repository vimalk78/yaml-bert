"""Plot per-epoch training loss from a training.log.

Parses lines of the form:
    Epoch N/M — loss: 1.234 (kind: 0.567 | simple: 0.890)

Emits an interactive HTML line plot with three traces (total, simple, kind).

Usage:
    python scripts/plot_training_loss.py output_v6.1_lever1_only_seed42/training.log
    python scripts/plot_training_loss.py log1.log log2.log --labels v5 v6.1 --out docs/figures/loss.html
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import re

import plotly.graph_objects as go


_EPOCH_RE = re.compile(
    r"Epoch\s+(\d+)/\d+\s+—\s+loss:\s+([\d.]+)\s+\(kind:\s+([\d.]+)\s+\|\s+simple:\s+([\d.]+)\)"
)


def parse_log(path: str) -> dict[str, list[float]]:
    epochs: list[int] = []
    total: list[float] = []
    kind: list[float] = []
    simple: list[float] = []
    with open(path) as f:
        for line in f:
            m = _EPOCH_RE.search(line)
            if not m:
                continue
            epochs.append(int(m.group(1)))
            total.append(float(m.group(2)))
            kind.append(float(m.group(3)))
            simple.append(float(m.group(4)))
    return {"epochs": epochs, "total": total, "kind": kind, "simple": simple}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot training-loss curves")
    parser.add_argument("logs", nargs="+", help="Path(s) to training.log file(s)")
    parser.add_argument("--labels", nargs="+", default=None,
                        help="Label per log file (defaults to filename)")
    parser.add_argument("--out", type=str, default="docs/figures/training_loss.html")
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=800)
    args = parser.parse_args()

    runs = []
    labels = args.labels or [None] * len(args.logs)
    if len(labels) != len(args.logs):
        raise SystemExit("--labels must have the same number of values as logs")

    for log_path, label in zip(args.logs, labels):
        data = parse_log(log_path)
        if not data["epochs"]:
            print(f"Warning: no epoch lines found in {log_path}")
            continue
        runs.append((label or log_path, data))
        print(f"{label or log_path}: {len(data['epochs'])} epochs, "
              f"final loss {data['total'][-1]:.4f}")

    fig = go.Figure()
    palette_total = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]
    palette_simple = ["#6baed6", "#fc9272", "#a1d99b", "#bcbddc"]
    palette_kind = ["#08519c", "#a50f15", "#006d2c", "#54278f"]

    for i, (label, data) in enumerate(runs):
        suffix = f" — {label}" if len(runs) > 1 else ""
        fig.add_trace(go.Scatter(
            x=data["epochs"], y=data["total"],
            mode="lines+markers",
            name=f"total{suffix}",
            line=dict(color=palette_total[i % len(palette_total)], width=3),
            marker=dict(size=8),
            hovertemplate="epoch %{x}<br>total: %{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=data["epochs"], y=data["simple"],
            mode="lines+markers",
            name=f"simple_head{suffix}",
            line=dict(color=palette_simple[i % len(palette_simple)], width=2, dash="dash"),
            marker=dict(size=6),
            hovertemplate="epoch %{x}<br>simple: %{y:.4f}<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=data["epochs"], y=data["kind"],
            mode="lines+markers",
            name=f"kind_head{suffix}",
            line=dict(color=palette_kind[i % len(palette_kind)], width=2, dash="dot"),
            marker=dict(size=6),
            hovertemplate="epoch %{x}<br>kind: %{y:.4f}<extra></extra>",
        ))

    title = "Training loss per epoch"
    if len(runs) > 1:
        title += f" — {' vs '.join(label for label, _ in runs)}"
    fig.update_layout(
        title=title,
        xaxis_title="Epoch",
        yaxis_title="Loss (cross-entropy)",
        width=args.width, height=args.height,
        template="plotly_white",
        hovermode="x unified",
    )
    fig.write_html(args.out, include_plotlyjs="inline", config={"responsive": True})
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
