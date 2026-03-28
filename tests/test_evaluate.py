import glob
import os
import random

import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.dataset import YamlDataset
from yaml_bert.evaluate import YamlBertEvaluator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder, Vocabulary

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "k8s-yamls"
)

TEST_CONFIG: YamlBertConfig = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)


def _build_trained_model() -> tuple[YamlBertModel, YamlDataset, Vocabulary]:
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    all_nodes = []
    for path in glob.glob(os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True):
        nodes = linearizer.linearize_file(path)
        annotator.annotate(nodes)
        all_nodes.extend(nodes)

    vocab = VocabBuilder().build(all_nodes)

    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
        config=TEST_CONFIG,
    )

    emb = YamlBertEmbedding(
        config=TEST_CONFIG,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
    )
    model = YamlBertModel(
        config=TEST_CONFIG,
        embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
    )

    from yaml_bert.trainer import YamlBertTrainer
    train_config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2, num_epochs=3)
    trainer = YamlBertTrainer(config=train_config, model=model, dataset=dataset)
    trainer.train()

    return model, dataset, vocab


def test_evaluator_prediction_accuracy():
    model, dataset, vocab = _build_trained_model()
    evaluator = YamlBertEvaluator(model=model, dataset=dataset, vocab=vocab)

    results = evaluator.evaluate_prediction_accuracy()

    assert "top1_accuracy" in results
    assert "top5_accuracy" in results
    assert 0.0 <= results["top1_accuracy"] <= 1.0
    assert 0.0 <= results["top5_accuracy"] <= 1.0
    assert results["top5_accuracy"] >= results["top1_accuracy"]


def test_evaluator_embedding_analysis():
    model, dataset, vocab = _build_trained_model()
    evaluator = YamlBertEvaluator(model=model, dataset=dataset, vocab=vocab)

    results = evaluator.analyze_embeddings()

    assert len(results) > 0
    for entry in results:
        assert "key" in entry
        assert "position_a" in entry
        assert "position_b" in entry
        assert "cosine_similarity" in entry
        assert -1.0 <= entry["cosine_similarity"] <= 1.0


def test_evaluator_top_k_predictions():
    model, dataset, vocab = _build_trained_model()
    evaluator = YamlBertEvaluator(model=model, dataset=dataset, vocab=vocab)

    random.seed(42)
    predictions = evaluator.top_k_predictions(doc_idx=0, k=5)

    assert len(predictions) > 0
    for pred in predictions:
        assert "position" in pred
        assert "true_key" in pred
        assert "predicted_keys" in pred
        assert len(pred["predicted_keys"]) <= 5
