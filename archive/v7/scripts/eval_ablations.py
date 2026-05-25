"""Evaluate trained ablation checkpoints and print a comparison table.

For each output dir, loads the final checkpoint (auto-detects the variant from
the checkpoint itself), runs the full capability-test suite, and reports
parameter count + pass rate per (variant, seed).

Usage:
    python scripts/eval_ablations.py output_ablation_full_seed42 output_ablation_no_depth_seed42 ...
    python scripts/eval_ablations.py output_ablation_*
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import glob
import os
import re
import sys

import torch

# Allow `from model_tests...` imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_tests"))
from test_capabilities import build_capabilities, run_test  # type: ignore

from yaml_bert.config import TreePosVariant, YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.vocab import Vocabulary


def find_last_checkpoint(output_dir: str) -> str:
    ckpts = glob.glob(os.path.join(output_dir, "checkpoints", "yaml_bert_v4_epoch_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints in {output_dir}/checkpoints/")

    def epoch_num(p: str) -> int:
        m = re.search(r"epoch_(\d+)\.pt$", p)
        return int(m.group(1)) if m else -1

    return max(ckpts, key=epoch_num)


def load_for_eval(output_dir: str) -> tuple[YamlBertModel, Vocabulary, TreePosVariant, int]:
    """Load model with the variant recorded in its own checkpoint."""
    vocab_path = os.path.join(output_dir, "vocab.json")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(vocab_path)
    vocab = Vocabulary.load(vocab_path)

    checkpoint_path = find_last_checkpoint(output_dir)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Older checkpoints (v5) don't store the variant — assume FULL.
    variant_str = checkpoint.get("tree_pos_variant", TreePosVariant.FULL.value)
    variant = TreePosVariant(variant_str)
    config = YamlBertConfig(tree_pos_variant=variant)

    torch.manual_seed(42)
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
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, vocab, variant, checkpoint.get("epoch", -1)


def evaluate_one(output_dir: str, phase_filter: str | None = None) -> dict:
    model, vocab, variant, epoch = load_for_eval(output_dir)
    n_params = sum(p.numel() for p in model.parameters())

    capabilities = build_capabilities()
    total = passed = 0
    per_cap: list[tuple[str, int, int]] = []
    for cap in capabilities:
        if phase_filter and cap.phase != phase_filter:
            continue
        cap_passed = cap_total = 0
        for test in cap.tests:
            res = run_test(model, vocab, test)
            cap_total += 1
            if res.passed:
                cap_passed += 1
        total += cap_total
        passed += cap_passed
        per_cap.append((cap.name, cap_passed, cap_total))

    return {
        "dir": output_dir,
        "variant": variant.value,
        "epoch": epoch,
        "n_params": n_params,
        "passed": passed,
        "total": total,
        "per_cap": per_cap,
    }


def seed_from_dir(name: str) -> str:
    m = re.search(r"seed(\d+)", name)
    return m.group(1) if m else "?"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ablation checkpoints")
    parser.add_argument("output_dirs", nargs="+", help="Per-variant output dirs")
    parser.add_argument(
        "--phase",
        default="pretrain",
        choices=["pretrain", "finetune", "all"],
        help="Capability phase filter (default: pretrain)",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    phase_filter = None if args.phase == "all" else args.phase

    results = []
    for d in args.output_dirs:
        if not os.path.isdir(d):
            print(f"SKIP (not a dir): {d}", file=sys.stderr)
            continue
        print(f"Evaluating {d} ...", file=sys.stderr)
        try:
            r = evaluate_one(d, phase_filter=phase_filter)
        except Exception as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            continue
        results.append(r)
        print(
            f"  {r['variant']:11s} epoch={r['epoch']}  params={r['n_params']:,}  "
            f"capability={r['passed']}/{r['total']}",
            file=sys.stderr,
        )

    if not results:
        print("No results.", file=sys.stderr)
        sys.exit(1)

    # Markdown table
    print()
    print("| Variant | Seed | Epoch | Params | Capability tests (pretrain) | Pass rate |")
    print("|---|---|---|---|---|---|")
    for r in sorted(results, key=lambda x: (x["variant"], seed_from_dir(x["dir"]))):
        rate = r["passed"] / r["total"] if r["total"] else 0.0
        print(
            f"| {r['variant']} | {seed_from_dir(r['dir'])} | {r['epoch']} | "
            f"{r['n_params']:,} | {r['passed']}/{r['total']} | {rate:.1%} |"
        )

    if args.verbose:
        print()
        for r in results:
            print(f"\n### {r['dir']}")
            for name, p, t in r["per_cap"]:
                marker = "✓" if p == t else "✗"
                print(f"  {marker} {name}: {p}/{t}")


if __name__ == "__main__":
    main()
