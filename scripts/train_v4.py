"""Train YAML-BERT v4 with hybrid bigram/trigram targets.

Usage:
    python scripts/train_v4.py --max-docs 5000 --epochs 10 --output-dir output_v4_quick
    python scripts/train_v4.py --max-docs 0 --epochs 15 --vocab-min-freq 100 --output-dir output_v4
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
import time

from yaml_bert.cache import build_or_load_cache
from yaml_bert.config import YamlBertConfig
from yaml_bert.dataset import YamlDataset
from yaml_bert.embedding import YamlBertEmbeddingV4
from yaml_bert.model import YamlBertModelV4
from yaml_bert.trainer import YamlBertTrainerV4
from yaml_bert.vocab import VocabBuilder, Vocabulary

DATASET_NAME = "substratusai/the-stack-yaml-k8s"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YAML-BERT v4")
    parser.add_argument("--max-docs", type=int, default=5000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--vocab-min-freq", type=int, default=10)
    parser.add_argument("--output-dir", type=str, default="output_v4")
    parser.add_argument("--resume", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    max_docs = None if args.max_docs == 0 else args.max_docs
    config = YamlBertConfig(
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    # Step 0: Linearize (cached + parallel)
    cache_path = os.path.join(args.output_dir, "doc_cache.pkl")
    print("=" * 60)
    print("Step 0: Linearizing documents")
    print("=" * 60)
    cached_docs = build_or_load_cache(DATASET_NAME, cache_path=cache_path, max_docs=max_docs)

    # Step 1: Build vocabulary
    vocab_path = os.path.join(args.output_dir, "vocab.json")
    print("\n" + "=" * 60)
    print("Step 1: Building vocabulary (with hybrid targets)")
    print("=" * 60)
    all_nodes = []
    for doc in cached_docs:
        all_nodes.extend(doc)
    vocab = VocabBuilder().build(all_nodes, min_freq=args.vocab_min_freq)
    vocab.save(vocab_path)

    print(f"Key vocab: {len(vocab.key_vocab)} tokens")
    print(f"Value vocab: {len(vocab.value_vocab)} tokens")
    print(f"Simple target vocab: {vocab.simple_target_vocab_size} targets")
    print(f"Kind target vocab: {vocab.kind_target_vocab_size} targets")

    # Step 2: Build v4 dataset
    print("\n" + "=" * 60)
    print("Step 2: Building v4 dataset")
    print("=" * 60)
    start = time.time()
    dataset = YamlDataset.from_cached_docs_v4(cached_docs, vocab, config)
    print(f"Dataset: {len(dataset)} documents ({time.time()-start:.1f}s)")

    # Step 3: Build v4 model
    print("\n" + "=" * 60)
    print("Step 3: Building v4 model")
    print("=" * 60)
    emb = YamlBertEmbeddingV4(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModelV4(
        config=config,
        embedding=emb,
        simple_vocab_size=vocab.simple_target_vocab_size,
        kind_vocab_size=vocab.kind_target_vocab_size,
    )

    # Step 4: Train
    print("\n" + "=" * 60)
    print("Step 4: Training v4")
    print("=" * 60)
    trainer = YamlBertTrainerV4(
        config=config,
        model=model,
        dataset=dataset,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        checkpoint_every=1,
        resume_from=args.resume,
    )
    losses = trainer.train()
    print(f"\nTraining complete. Final loss: {losses[-1]:.4f}")


if __name__ == "__main__":
    main()
