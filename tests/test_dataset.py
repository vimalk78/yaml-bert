import os

import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.dataset import YamlDataset
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder
from yaml_bert.types import NodeType

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "data", "k8s-yamls"
)


def _build_vocab():
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    import glob
    all_nodes = []
    for path in glob.glob(os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True):
        nodes = linearizer.linearize_file(path)
        annotator.annotate(nodes)
        all_nodes.extend(nodes)
    return VocabBuilder().build(all_nodes)


def test_dataset_length():
    vocab = _build_vocab()
    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
    )
    assert len(dataset) > 40


def test_dataset_item_keys():
    vocab = _build_vocab()
    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
    )
    item = dataset[0]

    expected_keys = {
        "token_ids", "node_types", "depths",
        "sibling_indices", "parent_key_ids", "labels",
    }
    assert set(item.keys()) == expected_keys

    seq_len = item["token_ids"].shape[0]
    for key in expected_keys:
        assert item[key].shape == (seq_len,), f"{key} shape mismatch"

    for key in expected_keys:
        assert item[key].dtype == torch.long, f"{key} dtype mismatch"


def test_dataset_masking_only_keys():
    vocab = _build_vocab()
    high_mask_config = YamlBertConfig(mask_prob=0.5)
    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
        config=high_mask_config,
    )

    item = dataset[0]
    labels = item["labels"]
    node_types = item["node_types"]

    masked_positions = labels != -100
    if masked_positions.any():
        masked_types = node_types[masked_positions]
        for t in masked_types:
            assert t.item() in (0, 2), f"Masked a non-key node: type={t.item()}"
