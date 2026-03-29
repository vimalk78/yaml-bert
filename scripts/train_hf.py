"""Train YAML-BERT on HuggingFace K8s YAML dataset.

Usage:
    python train_hf.py                          # 1000 docs, small model (quick test)
    python train_hf.py --max-docs 10000         # 10K docs
    python train_hf.py --max-docs 0 --full      # all 276K docs, full model config
    python train_hf.py --resume output_hf/checkpoints/yaml_bert_epoch_10.pt
"""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import os
import time

from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import YamlDataset
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.evaluate import YamlBertEvaluator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.trainer import YamlBertTrainer
from yaml_bert.visualize import plot_training_loss, plot_embedding_similarity, plot_accuracy
from yaml_bert.vocab import VocabBuilder, Vocabulary


DATASET_NAME: str = "substratusai/the-stack-yaml-k8s"
_PROJECT_ROOT: str = os.path.join(os.path.dirname(__file__), "..")
OUTPUT_DIR: str = os.path.join(_PROJECT_ROOT, "output_hf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YAML-BERT on HuggingFace K8s YAML dataset")
    parser.add_argument("--max-docs", type=int, default=1000,
                        help="Max documents to load (0 = all, default: 1000)")
    parser.add_argument("--full", action="store_true",
                        help="Use full model config (d_model=256, 6 layers, 8 heads)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override batch size")
    parser.add_argument("--output-dir", type=str, default=OUTPUT_DIR,
                        help="Output directory")
    parser.add_argument("--vocab-min-freq", type=int, default=2,
                        help="Min frequency for vocab tokens (default: 2)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    max_docs: int | None = None if args.max_docs == 0 else args.max_docs

    # Select config
    if args.full:
        config = YamlBertConfig()  # full defaults: d_model=256, 6 layers, 8 heads
    else:
        config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)

    if args.epochs is not None:
        config.num_epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size

    # Use fewer epochs for small subsets, more for large
    if config.num_epochs == 30 and max_docs is not None:
        if max_docs <= 1000:
            config.num_epochs = 20
        elif max_docs <= 10000:
            config.num_epochs = 15

    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()

    # Step 1: Build or load vocabulary
    vocab_path: str = os.path.join(args.output_dir, "vocab.json")
    counts_path: str = os.path.join(args.output_dir, "token_counts.json")

    print("=" * 60)
    print("Step 1: Building vocabulary")
    print("=" * 60)
    builder: VocabBuilder = VocabBuilder()
    vocab: Vocabulary = builder.build_from_huggingface(
        DATASET_NAME,
        linearizer=linearizer,
        annotator=annotator,
        max_docs=max_docs,
        min_freq=args.vocab_min_freq,
        counts_path=counts_path,
    )
    vocab.save(vocab_path)

    print(f"Key vocab: {len(vocab.key_vocab)} tokens")
    print(f"Value vocab: {len(vocab.value_vocab)} tokens")

    # Step 2: Load dataset
    print("\n" + "=" * 60)
    print("Step 2: Loading dataset")
    print("=" * 60)
    start: float = time.time()
    dataset: YamlDataset = YamlDataset.from_huggingface(
        DATASET_NAME,
        vocab=vocab,
        linearizer=linearizer,
        annotator=annotator,
        config=config,
        max_docs=max_docs,
    )
    load_time: float = time.time() - start
    print(f"Dataset: {len(dataset):,} documents (loaded in {load_time:.1f}s)")

    # Step 3: Build model
    print("\n" + "=" * 60)
    print("Step 3: Building model")
    print("=" * 60)
    emb: YamlBertEmbedding = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model: YamlBertModel = YamlBertModel(
        config=config,
        embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
    )

    # Step 4: Train
    print("\n" + "=" * 60)
    print("Step 4: Training")
    print("=" * 60)
    trainer: YamlBertTrainer = YamlBertTrainer(
        config=config,
        model=model,
        dataset=dataset,
        checkpoint_dir=os.path.join(args.output_dir, "checkpoints"),
        checkpoint_every=5,
        resume_from=args.resume,
    )

    start = time.time()
    losses: list[float] = trainer.train()
    train_time: float = time.time() - start
    print(f"\nTraining completed in {train_time:.1f}s ({train_time/60:.1f} min)")

    # Step 5: Visualize
    print("\n" + "=" * 60)
    print("Step 5: Saving visualizations")
    print("=" * 60)
    plot_training_loss(
        losses,
        output_path=os.path.join(args.output_dir, "training_loss.png"),
    )

    # Step 6: Evaluate
    print("\n" + "=" * 60)
    print("Step 6: Evaluating")
    print("=" * 60)
    evaluator: YamlBertEvaluator = YamlBertEvaluator(
        model=model, dataset=dataset, vocab=vocab,
    )

    accuracy = evaluator.evaluate_prediction_accuracy()
    print(f"Top-1 accuracy: {accuracy['top1_accuracy']:.2%}")
    print(f"Top-5 accuracy: {accuracy['top5_accuracy']:.2%}")
    print(f"Total masked: {accuracy['total_masked']}")

    plot_accuracy(
        accuracy,
        output_path=os.path.join(args.output_dir, "accuracy.png"),
    )

    embeddings = evaluator.analyze_embeddings()
    for entry in embeddings:
        print(
            f"  {entry['key']}: "
            f"({entry['position_a']}) vs ({entry['position_b']}) "
            f"cosine_sim={entry['cosine_similarity']:.4f}"
        )

    plot_embedding_similarity(
        embeddings,
        output_path=os.path.join(args.output_dir, "embedding_similarity.png"),
    )

    # Step 7: Sample predictions
    print("\n" + "=" * 60)
    print("Step 7: Sample predictions")
    print("=" * 60)
    for doc_idx in range(min(3, len(dataset))):
        predictions = evaluator.top_k_predictions(doc_idx=doc_idx, k=5)
        if predictions:
            print(f"\nDocument {doc_idx}:")
            for pred in predictions[:3]:  # show first 3 masked positions
                print(f"  Position {pred['position']}: true='{pred['true_key']}'")
                for i, pk in enumerate(pred["predicted_keys"]):
                    marker = " <--" if pk["key"] == pred["true_key"] else ""
                    print(f"    {i+1}. '{pk['key']}' ({pk['probability']:.2%}){marker}")

    print("\n" + "=" * 60)
    print(f"Done! Outputs saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
