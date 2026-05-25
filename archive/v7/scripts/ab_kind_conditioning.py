"""A/B comparison: kind-conditioning ON vs OFF.

Runs suggest_missing_fields on a handful of representative YAMLs twice
(mask on, mask off) and diffs the outputs to settle whether
kind-conditioning is doing material work.

Usage:
    python scripts/ab_kind_conditioning.py output_v6.1_lever1_only_seed42/checkpoints/yaml_bert_v4_epoch_30.pt
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import ast
import os

import torch

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.suggest import suggest_missing_fields
from yaml_bert.vocab import Vocabulary


def load_app_examples(app_path: str) -> dict[str, str]:
    """Extract EXAMPLE_* string constants from app.py without importing it
    (importing would trigger torch + model load)."""
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


def load_model(checkpoint_path: str, vocab_path: str) -> tuple[YamlBertModel, Vocabulary]:
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


def as_key_set(suggestions: list[dict]) -> set[tuple[str, str]]:
    return {(s["parent_path"], s["missing_key"]) for s in suggestions}


def as_conf_map(suggestions: list[dict]) -> dict[tuple[str, str], float]:
    return {(s["parent_path"], s["missing_key"]): s["confidence"] for s in suggestions}


def split_docs(yaml_text: str) -> list[str]:
    """Mirror app.py's multi-doc split. Returns single-doc YAML strings."""
    import re
    parts = re.split(r'^---\s*$', yaml_text, flags=re.MULTILINE)
    return [p.strip() + "\n" for p in parts if p.strip()]


def compare(name: str, yaml_text: str, model: YamlBertModel, vocab: Vocabulary, threshold: float) -> dict:
    # multi-doc YAMLs: aggregate across all docs
    all_on, all_off = [], []
    with torch.no_grad():
        for doc in split_docs(yaml_text):
            on_suggs, _ = suggest_missing_fields(
                model, vocab, doc, threshold=threshold, kind_conditioning=True
            )
            off_suggs, _ = suggest_missing_fields(
                model, vocab, doc, threshold=threshold, kind_conditioning=False
            )
            all_on.extend(on_suggs)
            all_off.extend(off_suggs)
    on_suggs, off_suggs = all_on, all_off

    on_set = as_key_set(on_suggs)
    off_set = as_key_set(off_suggs)
    on_conf = as_conf_map(on_suggs)
    off_conf = as_conf_map(off_suggs)

    only_on = on_set - off_set
    only_off = off_set - on_set
    shared = on_set & off_set

    conf_deltas = [
        (parent, key, on_conf[(parent, key)] - off_conf[(parent, key)])
        for (parent, key) in shared
    ]
    conf_deltas.sort(key=lambda t: -abs(t[2]))

    print(f"\n{'=' * 70}")
    print(f"  {name}")
    print(f"{'=' * 70}")
    print(f"  ON  : {len(on_suggs):3d} suggestions")
    print(f"  OFF : {len(off_suggs):3d} suggestions")
    print(f"  shared: {len(shared)} | only-ON: {len(only_on)} | only-OFF: {len(only_off)}")

    if only_on:
        print(f"\n  Appearing ONLY with mask ON (kind-conditioning surfaced these):")
        for parent, key in sorted(only_on):
            path = parent if parent else "(root)"
            print(f"    + {path}.{key}  conf={on_conf[(parent, key)]:.1%}")

    if only_off:
        print(f"\n  Appearing ONLY with mask OFF (kind-conditioning suppressed these):")
        for parent, key in sorted(only_off):
            path = parent if parent else "(root)"
            print(f"    - {path}.{key}  conf={off_conf[(parent, key)]:.1%}")

    if conf_deltas:
        nonzero = [(p, k, d) for (p, k, d) in conf_deltas if abs(d) > 0.001]
        if nonzero:
            print(f"\n  Confidence shifts on shared keys (top 5 by magnitude):")
            for parent, key, delta in nonzero[:5]:
                path = parent if parent else "(root)"
                arrow = "↑" if delta > 0 else "↓"
                print(f"    {arrow} {path}.{key}  {off_conf[(parent, key)]:.1%} → {on_conf[(parent, key)]:.1%}  (Δ={delta:+.1%})")

    return {
        "name": name,
        "on_count": len(on_suggs),
        "off_count": len(off_suggs),
        "only_on": len(only_on),
        "only_off": len(only_off),
        "max_conf_delta": max((abs(d) for _, _, d in conf_deltas), default=0.0),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--vocab", default=None)
    parser.add_argument("--threshold", type=float, default=0.1,
                        help="Confidence threshold (default 0.1, matches app default)")
    parser.add_argument("--app", default="app.py",
                        help="Path to app.py to pull EXAMPLE_* from (default app.py in cwd)")
    args = parser.parse_args()

    vocab_path = args.vocab
    if vocab_path is None:
        ckpt_dir = os.path.dirname(args.checkpoint)
        for cand in [os.path.join(ckpt_dir, "vocab.json"), os.path.join(ckpt_dir, "..", "vocab.json")]:
            if os.path.exists(cand):
                vocab_path = cand
                break
        if vocab_path is None:
            print("Could not find vocab.json. Pass --vocab.")
            return

    print(f"Loading model from {args.checkpoint}")
    print(f"Vocab from {vocab_path}")
    print(f"Examples from {args.app}")
    print(f"Threshold: {args.threshold}")
    model, vocab = load_model(args.checkpoint, vocab_path)

    SAMPLES = load_app_examples(args.app)
    if not SAMPLES:
        print(f"No EXAMPLE_* constants found in {args.app}")
        return
    print(f"Loaded {len(SAMPLES)} examples from app")

    summaries = []
    for name, yaml_text in SAMPLES.items():
        summaries.append(compare(name, yaml_text, model, vocab, args.threshold))

    print(f"\n{'=' * 70}")
    print(f"  Summary")
    print(f"{'=' * 70}")
    print(f"  {'sample':<28} {'ON':>4} {'OFF':>4} {'+ON':>4} {'-ON':>4} {'maxΔ':>6}")
    for s in summaries:
        print(f"  {s['name']:<28} {s['on_count']:>4} {s['off_count']:>4} {s['only_on']:>4} {s['only_off']:>4} {s['max_conf_delta']:>6.1%}")

    total_diff = sum(s["only_on"] + s["only_off"] for s in summaries)
    max_delta = max(s["max_conf_delta"] for s in summaries)
    print()
    if total_diff == 0 and max_delta < 0.01:
        print("  VERDICT: kind-conditioning is doing NOTHING measurable.")
        print("           (no different suggestions, no confidence shifts > 1%)")
    elif total_diff == 0:
        print(f"  VERDICT: kind-conditioning only shifts CONFIDENCES (max Δ={max_delta:.1%}),")
        print("           but never changes which suggestions appear above threshold.")
    else:
        print(f"  VERDICT: kind-conditioning materially changes outputs:")
        print(f"           {total_diff} suggestion membership differences, max Δ={max_delta:.1%}.")


if __name__ == "__main__":
    main()
