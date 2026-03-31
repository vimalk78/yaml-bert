"""Suggest missing fields in Kubernetes YAML using YAML-BERT conventions.

Usage:
    python scripts/suggest_fields.py output_v4/yaml_bert_final.pt --yaml-file my-deployment.yaml
    python scripts/suggest_fields.py output_v4/yaml_bert_final.pt --yaml-file my-pod.yaml --threshold 0.5
    python scripts/suggest_fields.py output_v4/yaml_bert_final.pt --yaml-dir ./manifests/
    python scripts/suggest_fields.py output_v4/yaml_bert_final.pt --yaml-file my-pod.yaml --format json
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import glob
import json
import os

import torch

from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.suggest import suggest_missing_fields
from yaml_bert.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Suggest missing fields in K8s YAML")
    parser.add_argument("checkpoint", type=str, help="Path to model checkpoint")
    parser.add_argument("--vocab", type=str, default=None, help="Vocab path (auto-detected if not set)")
    parser.add_argument("--yaml-file", type=str, default=None, help="Single YAML file")
    parser.add_argument("--yaml-dir", type=str, default=None, help="Directory of YAML files")
    parser.add_argument("--yaml-text", type=str, default=None, help="Inline YAML text")
    parser.add_argument("--threshold", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--format", type=str, choices=["text", "json"], default="text")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def load_model(checkpoint_path: str, vocab_path: str, device: str) -> tuple[YamlBertModel, Vocabulary]:
    vocab: Vocabulary = Vocabulary.load(vocab_path)
    config: YamlBertConfig = YamlBertConfig()
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
    checkpoint: dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, vocab


def print_report(suggestions: list[dict], source: str = "", fmt: str = "text") -> None:
    if fmt == "json":
        print(json.dumps({"source": source, "suggestions": suggestions}, indent=2))
        return

    if source:
        print(f"\n{'=' * 60}")
        print(f"  {source}")
        print(f"{'=' * 60}")

    if not suggestions:
        print("  No missing fields detected.")
        return

    by_parent: dict[str, list[dict]] = {}
    for s in suggestions:
        by_parent.setdefault(s["parent_path"], []).append(s)

    for parent, items in by_parent.items():
        path_display: str = parent if parent else "(root)"
        print(f"\n  {path_display}:")
        for item in items:
            conf: float = item["confidence"]
            strength: str = "STRONG" if conf > 0.8 else "MODERATE" if conf > 0.5 else "WEAK"
            print(f"    [{conf:5.1%}] {item['missing_key']} ({strength})")

    print(f"\n  Total: {len(suggestions)} suggestions")


def main() -> None:
    args = parse_args()

    vocab_path: str | None = args.vocab
    if vocab_path is None:
        checkpoint_dir: str = os.path.dirname(args.checkpoint)
        for candidate in [
            os.path.join(checkpoint_dir, "vocab.json"),
            os.path.join(checkpoint_dir, "..", "vocab.json"),
        ]:
            if os.path.exists(candidate):
                vocab_path = candidate
                break
        if vocab_path is None:
            print("Error: could not find vocab.json. Specify --vocab.")
            return

    print(f"Loading model...")
    model, vocab = load_model(args.checkpoint, vocab_path, args.device)

    yaml_files: list[tuple[str, str]] = []

    if args.yaml_file:
        with open(args.yaml_file) as f:
            yaml_files.append((args.yaml_file, f.read()))
    elif args.yaml_dir:
        for path in sorted(glob.glob(os.path.join(args.yaml_dir, "**", "*.yaml"), recursive=True)):
            with open(path) as f:
                yaml_files.append((path, f.read()))
    elif args.yaml_text:
        yaml_files.append(("(inline)", args.yaml_text))
    else:
        print("Provide --yaml-file, --yaml-dir, or --yaml-text")
        return

    import yaml as pyyaml
    for source, yaml_text in yaml_files:
        # Split multi-document YAML on '---'
        documents: list[str] = [
            doc.strip() for doc in yaml_text.split("\n---\n")
            if doc.strip() and not doc.strip().startswith("#")
        ]
        # If no split happened, treat the whole text as one document
        if not documents:
            documents = [yaml_text]

        for doc_text in documents:
            doc_source: str = source
            try:
                doc = pyyaml.safe_load(doc_text)
                if isinstance(doc, dict):
                    kind: str = doc.get("kind", "")
                    name: str = doc.get("metadata", {}).get("name", "") if isinstance(doc.get("metadata"), dict) else ""
                    if kind:
                        doc_source = f"{source} ({kind}/{name})" if name else f"{source} ({kind})"
            except Exception:
                pass

            suggestions = suggest_missing_fields(model, vocab, doc_text, threshold=args.threshold)
            print_report(suggestions, source=doc_source, fmt=args.format)

    if len(yaml_files) > 1 and args.format == "text":
        total = sum(
            len(suggest_missing_fields(model, vocab, yt, threshold=args.threshold))
            for _, yt in yaml_files
        )
        print(f"\n{'=' * 60}")
        print(f"  Total: {total} suggestions across {len(yaml_files)} files")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
