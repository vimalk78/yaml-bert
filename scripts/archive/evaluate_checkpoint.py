"""Evaluate a saved YAML-BERT checkpoint without interrupting training."""
from __future__ import annotations
import _setup_path  # noqa: F401

import argparse
import glob
import os

import torch

from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import YamlDataset
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.evaluate import YamlBertEvaluator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.visualize import plot_accuracy, plot_embedding_similarity
from yaml_bert.vocab import Vocabulary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YAML-BERT checkpoint")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint .pt file")
    parser.add_argument("--vocab", type=str, default="output_v1/vocab.json")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Local YAML dir (if not using HF dataset)")
    parser.add_argument("--hf-dataset", type=str, default="substratusai/the-stack-yaml-k8s",
                        help="HuggingFace dataset name")
    parser.add_argument("--max-eval-docs", type=int, default=500,
                        help="Max docs to evaluate on (default: 500)")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device for evaluation (default: cpu, to not compete with training GPU)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("Loading vocabulary...")
    vocab: Vocabulary = Vocabulary.load(args.vocab)
    print(f"Key vocab: {len(vocab.key_vocab)}, Value vocab: {len(vocab.value_vocab)}")

    config: YamlBertConfig = YamlBertConfig()

    print("Loading evaluation dataset...")
    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()

    if args.data_dir:
        dataset: YamlDataset = YamlDataset(
            yaml_dir=args.data_dir,
            vocab=vocab,
            linearizer=linearizer,
            annotator=annotator,
            config=config,
        )
    else:
        dataset = YamlDataset.from_huggingface(
            args.hf_dataset,
            vocab=vocab,
            linearizer=linearizer,
            annotator=annotator,
            config=config,
            max_docs=args.max_eval_docs,
        )
    print(f"Evaluation dataset: {len(dataset)} documents")

    print("Building model and loading checkpoint...")
    emb: YamlBertEmbedding = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    model: YamlBertModel = YamlBertModel(
        config=config,
        embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )

    checkpoint: dict = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.to(args.device)
    print(f"Loaded checkpoint: epoch {checkpoint['epoch']}")

    evaluator: YamlBertEvaluator = YamlBertEvaluator(
        model=model, dataset=dataset, vocab=vocab,
    )

    # Accuracy
    print("\n" + "=" * 60)
    print("Prediction Accuracy")
    print("=" * 60)
    accuracy = evaluator.evaluate_prediction_accuracy()
    print(f"Top-1: {accuracy['top1_accuracy']:.2%}")
    print(f"Top-5: {accuracy['top5_accuracy']:.2%}")
    print(f"Total masked: {accuracy['total_masked']}")

    output_dir: str = os.path.dirname(args.checkpoint)
    plot_accuracy(accuracy, output_path=os.path.join(output_dir, "accuracy.png"))

    # Embedding analysis
    print("\n" + "=" * 60)
    print("Embedding Analysis")
    print("=" * 60)
    embeddings = evaluator.analyze_embeddings()
    for entry in embeddings:
        print(
            f"  {entry['key']}: "
            f"({entry['position_a']}) vs ({entry['position_b']}) "
            f"cosine_sim={entry['cosine_similarity']:.4f}"
        )
    plot_embedding_similarity(embeddings, output_path=os.path.join(output_dir, "embedding_similarity.png"))

    # Sample predictions
    print("\n" + "=" * 60)
    print("Sample Predictions")
    print("=" * 60)
    for doc_idx in range(min(5, len(dataset))):
        predictions = evaluator.top_k_predictions(doc_idx=doc_idx, k=5)
        if predictions:
            print(f"\nDocument {doc_idx}:")
            for pred in predictions[:3]:
                print(f"  Position {pred['position']}: true='{pred['true_key']}'")
                for i, pk in enumerate(pred["predicted_keys"]):
                    marker = " <--" if pk["key"] == pred["true_key"] else ""
                    print(f"    {i+1}. '{pk['key']}' ({pk['probability']:.2%}){marker}")

    print("\nDone!")


if __name__ == "__main__":
    main()
