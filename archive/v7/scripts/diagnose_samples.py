"""Diagnose model behavior across all 12 app sample YAMLs.

For each sample and each probed parent, prints the top 5 raw candidates
with status (KEEP/EXIST/BELOW/MGMT/DROP). Captures the verbose output
from suggest_missing_fields() per sample.

Usage:
    python scripts/diagnose_samples.py output_v6.1_lever1_only_seed42/checkpoints/yaml_bert_v4_epoch_30.pt
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import ast
import contextlib
import io
import os
import re

import torch

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.suggest import suggest_missing_fields
from yaml_bert.vocab import Vocabulary


def load_app_examples(app_path: str) -> dict[str, str]:
    with open(app_path) as f:
        tree = ast.parse(f.read())
    samples: dict[str, str] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if (isinstance(target, ast.Name)
                    and target.id.startswith("EXAMPLE_")
                    and isinstance(node.value, ast.Constant)
                    and isinstance(node.value.value, str)):
                samples[target.id] = node.value.value
    return samples


def load_model(checkpoint_path: str, vocab_path: str):
    vocab = Vocabulary.load(vocab_path)
    config = YamlBertConfig()
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModel(
        config=config, embedding=emb,
        simple_vocab_size=vocab.simple_target_vocab_size,
        kind_vocab_size=vocab.kind_target_vocab_size,
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, vocab


def split_docs(yaml_text: str) -> list[str]:
    parts = re.split(r'^---\s*$', yaml_text, flags=re.MULTILINE)
    return [p.strip() + "\n" for p in parts if p.strip()]


# Parse the verbose output stream into per-parent records.
_PARENT_RE = re.compile(r"^\[([^\]]*)\]\s+depth=(\d+)\s+head=(\S+)")
_CAND_RE = re.compile(r"^\s+([✓·↓✗])\s+(\w+)\s+([\d.]+)%\s+(.+?)(?:\s+\((.+)\))?$")


def parse_verbose(text: str) -> list[dict]:
    blocks = []
    current = None
    for line in text.splitlines():
        m = _PARENT_RE.match(line)
        if m:
            if current:
                blocks.append(current)
            current = {
                "parent": m.group(1),
                "depth": int(m.group(2)),
                "head": m.group(3),
                "candidates": [],
            }
            continue
        if current is None:
            continue
        m = _CAND_RE.match(line)
        if m:
            current["candidates"].append({
                "status": m.group(2),
                "prob": float(m.group(3)) / 100.0,
                "target": m.group(4),
                "reason": m.group(5) or "",
            })
    if current:
        blocks.append(current)
    return blocks


def render_block(block: dict, top_n: int = 5) -> str:
    parent = block["parent"] or "(root)"
    head_short = "kind" if block["head"].startswith("kind") else "simple"
    out = [f"  [{parent}]  d={block['depth']}  {head_short}"]
    for c in block["candidates"][:top_n]:
        sym = {"KEEP": "✓", "EXIST": "·", "BELOW": "↓", "MGMT": "·", "DROP": "✗"}.get(c["status"], "?")
        target_short = c["target"].split("::")[-1] if "::" in c["target"] else c["target"]
        prefix = f"{c['target'].split('::')[-2]}::" if "::" in c["target"] else ""
        out.append(f"    {sym} {c['status']:>5} {c['prob']*100:6.2f}%  {prefix}{target_short}")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--threshold", type=float, default=0.1)
    parser.add_argument("--app", default="app.py")
    parser.add_argument("--top-n", type=int, default=5, help="Top-N candidates to show per parent")
    parser.add_argument("--samples", nargs="*", default=None,
                        help="Specific EXAMPLE_* names (default: all)")
    args = parser.parse_args()

    vocab_path = args.vocab
    if vocab_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        for cand in [os.path.join(ckpt_dir, "vocab.json"), os.path.join(ckpt_dir, "..", "vocab.json")]:
            if os.path.exists(cand):
                vocab_path = cand
                break

    model, vocab = load_model(args.checkpoint, vocab_path)
    SAMPLES = load_app_examples(args.app)

    if args.samples:
        SAMPLES = {k: v for k, v in SAMPLES.items() if k in args.samples}

    print(f"Threshold: {args.threshold}, top_n_shown: {args.top_n}")

    for name, yaml_text in SAMPLES.items():
        print(f"\n{'=' * 78}")
        print(f"  {name}")
        print(f"{'=' * 78}")

        for doc_i, doc in enumerate(split_docs(yaml_text), start=1):
            if len(split_docs(yaml_text)) > 1:
                print(f"\n--- doc {doc_i} ---")
            captured = io.StringIO()
            with torch.no_grad(), contextlib.redirect_stderr(captured):
                suggestions, _ = suggest_missing_fields(
                    model, vocab, doc,
                    threshold=args.threshold, verbose=True,
                )
            blocks = parse_verbose(captured.getvalue())
            for block in blocks:
                print(render_block(block, top_n=args.top_n))
            final = [s for s in suggestions]
            if final:
                print(f"\n  >> Final surfaced: {len(final)} suggestions")
                for s in sorted(final, key=lambda s: -s["confidence"]):
                    path = s["parent_path"] or "(root)"
                    print(f"     {s['confidence']*100:6.2f}%  {path}.{s['missing_key']}")
            else:
                print(f"\n  >> No suggestions above threshold {args.threshold}")


if __name__ == "__main__":
    main()
