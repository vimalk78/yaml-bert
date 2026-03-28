import glob
import os

import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.dataset import YamlDataset
from yaml_bert.trainer import YamlBertTrainer
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "k8s-yamls"
)

TEST_CONFIG: YamlBertConfig = YamlBertConfig(d_model=64, num_layers=2, num_heads=2, num_epochs=1)


def _build_model_and_dataset() -> tuple[YamlBertModel, YamlDataset]:
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

    return model, dataset


def test_trainer_runs_one_epoch():
    model, dataset = _build_model_and_dataset()

    trainer = YamlBertTrainer(
        config=TEST_CONFIG,
        model=model,
        dataset=dataset,
    )

    losses = trainer.train()

    assert len(losses) == 1
    assert losses[0] > 0


def test_trainer_loss_decreases():
    config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2, num_epochs=5)
    model, dataset = _build_model_and_dataset()

    trainer = YamlBertTrainer(
        config=config,
        model=model,
        dataset=dataset,
    )

    losses = trainer.train()

    assert len(losses) == 5
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: first={losses[0]:.4f}, last={losses[-1]:.4f}"
    )


def test_trainer_saves_checkpoint(tmp_path):
    model, dataset = _build_model_and_dataset()

    trainer = YamlBertTrainer(
        config=TEST_CONFIG,
        model=model,
        dataset=dataset,
        checkpoint_dir=str(tmp_path),
    )

    trainer.train()

    checkpoint_files = os.listdir(tmp_path)
    assert len(checkpoint_files) > 0
    assert any(f.endswith(".pt") for f in checkpoint_files)

    checkpoint_path = os.path.join(tmp_path, checkpoint_files[0])
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "epoch" in checkpoint


def test_trainer_resumes_from_checkpoint(tmp_path):
    model, dataset = _build_model_and_dataset()

    config_1 = YamlBertConfig(d_model=64, num_layers=2, num_heads=2, num_epochs=1)
    trainer1 = YamlBertTrainer(
        config=config_1,
        model=model,
        dataset=dataset,
        checkpoint_dir=str(tmp_path),
        checkpoint_every=1,
    )
    losses1 = trainer1.train()

    checkpoint_path = os.path.join(tmp_path, "yaml_bert_epoch_1.pt")
    config_2 = YamlBertConfig(d_model=64, num_layers=2, num_heads=2, num_epochs=2)
    trainer2 = YamlBertTrainer(
        config=config_2,
        model=model,
        dataset=dataset,
        resume_from=checkpoint_path,
    )
    losses2 = trainer2.train()

    assert len(losses2) == 1
