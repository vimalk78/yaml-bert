"""Dry run: train YAML-BERT on local K8s YAML corpus and evaluate."""
from __future__ import annotations

import glob
import os

from yaml_bert.config import YamlBertConfig
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.dataset import YamlDataset
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.evaluate import YamlBertEvaluator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.model import YamlBertModel
from yaml_bert.trainer import YamlBertTrainer
from yaml_bert.visualize import plot_training_loss, plot_embedding_similarity
from yaml_bert.vocab import VocabBuilder


DATA_DIR: str = os.path.join(os.path.dirname(__file__), "data", "k8s-yamls")
OUTPUT_DIR: str = os.path.join(os.path.dirname(__file__), "output")


def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Use small config for dry run on 52 files
    config: YamlBertConfig = YamlBertConfig(
        d_model=64,
        num_layers=2,
        num_heads=2,
        num_epochs=20,
        batch_size=8,
        lr=1e-3,
    )

    # Step 1: Build vocabulary from corpus
    print("=" * 60)
    print("Step 1: Building vocabulary")
    print("=" * 60)
    linearizer: YamlLinearizer = YamlLinearizer()
    annotator: DomainAnnotator = DomainAnnotator()

    all_nodes = []
    yaml_files = sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.yaml"), recursive=True))
    for path in yaml_files:
        nodes = linearizer.linearize_file(path)
        annotator.annotate(nodes)
        all_nodes.extend(nodes)

    vocab = VocabBuilder().build(all_nodes)
    vocab_path: str = os.path.join(OUTPUT_DIR, "vocab.json")
    vocab.save(vocab_path)
    print(f"Key vocab: {len(vocab.key_vocab)} tokens")
    print(f"Value vocab: {len(vocab.value_vocab)} tokens")
    print(f"Vocabulary saved: {vocab_path}")

    # Step 2: Create dataset
    print("\n" + "=" * 60)
    print("Step 2: Creating dataset")
    print("=" * 60)
    dataset: YamlDataset = YamlDataset(
        yaml_dir=DATA_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
        config=config,
    )
    print(f"Dataset: {len(dataset)} documents")

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
        checkpoint_dir=os.path.join(OUTPUT_DIR, "checkpoints"),
        checkpoint_every=10,
    )
    losses: list[float] = trainer.train()

    # Step 5: Visualize training loss
    print("\n" + "=" * 60)
    print("Step 5: Visualizing training loss")
    print("=" * 60)
    plot_training_loss(
        losses,
        output_path=os.path.join(OUTPUT_DIR, "training_loss.png"),
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

    embeddings = evaluator.analyze_embeddings()
    for entry in embeddings:
        print(
            f"  {entry['key']}: "
            f"({entry['position_a']}) vs ({entry['position_b']}) "
            f"cosine_sim={entry['cosine_similarity']:.4f}"
        )

    plot_embedding_similarity(
        embeddings,
        output_path=os.path.join(OUTPUT_DIR, "embedding_similarity.png"),
    )

    # Step 7: Show top-k predictions for first document
    print("\n" + "=" * 60)
    print("Step 7: Top-5 predictions (first document)")
    print("=" * 60)
    predictions = evaluator.top_k_predictions(doc_idx=0, k=5)
    for pred in predictions:
        print(f"\n  Position {pred['position']}: true key = '{pred['true_key']}'")
        for i, pk in enumerate(pred["predicted_keys"]):
            marker = " <--" if pk["key"] == pred["true_key"] else ""
            print(f"    {i+1}. '{pk['key']}' ({pk['probability']:.2%}){marker}")

    print("\n" + "=" * 60)
    print("Done! Outputs saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
