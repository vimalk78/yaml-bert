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
        "sibling_indices", "parent_key_ids", "labels", "kind_ids",
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


from yaml_bert.dataset import collate_fn


def test_collate_fn_padding():
    item1 = {
        "token_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "node_types": torch.tensor([0, 1, 0], dtype=torch.long),
        "depths": torch.tensor([0, 0, 1], dtype=torch.long),
        "sibling_indices": torch.tensor([0, 0, 0], dtype=torch.long),
        "parent_key_ids": torch.tensor([1, 1, 2], dtype=torch.long),
        "labels": torch.tensor([-100, 5, -100], dtype=torch.long),
    }
    item2 = {
        "token_ids": torch.tensor([4, 5], dtype=torch.long),
        "node_types": torch.tensor([0, 1], dtype=torch.long),
        "depths": torch.tensor([0, 0], dtype=torch.long),
        "sibling_indices": torch.tensor([0, 0], dtype=torch.long),
        "parent_key_ids": torch.tensor([1, 1], dtype=torch.long),
        "labels": torch.tensor([6, -100], dtype=torch.long),
    }

    batch = collate_fn([item1, item2])

    assert batch["token_ids"].shape == (2, 3)
    assert batch["padding_mask"].shape == (2, 3)

    assert batch["padding_mask"][0].tolist() == [False, False, False]
    assert batch["padding_mask"][1].tolist() == [False, False, True]

    assert batch["token_ids"][1, 2].item() == 0
    assert batch["labels"][1, 2].item() == -100


def test_dataset_includes_kind_ids():
    vocab = _build_vocab()
    dataset = YamlDataset(
        yaml_dir=TEMPLATES_DIR,
        vocab=vocab,
        linearizer=YamlLinearizer(),
        annotator=DomainAnnotator(),
    )
    item = dataset[0]

    assert "kind_ids" in item
    kind_ids = item["kind_ids"]
    assert kind_ids.shape == item["token_ids"].shape
    # All kind_ids should be the same within a document
    assert (kind_ids == kind_ids[0]).all()


def test_collate_fn_pads_kind_ids():
    item1 = {
        "token_ids": torch.tensor([1, 2, 3], dtype=torch.long),
        "node_types": torch.tensor([0, 1, 0], dtype=torch.long),
        "depths": torch.tensor([0, 0, 1], dtype=torch.long),
        "sibling_indices": torch.tensor([0, 0, 0], dtype=torch.long),
        "parent_key_ids": torch.tensor([1, 1, 2], dtype=torch.long),
        "labels": torch.tensor([-100, 5, -100], dtype=torch.long),
        "kind_ids": torch.tensor([3, 3, 3], dtype=torch.long),
    }
    item2 = {
        "token_ids": torch.tensor([4, 5], dtype=torch.long),
        "node_types": torch.tensor([0, 1], dtype=torch.long),
        "depths": torch.tensor([0, 0], dtype=torch.long),
        "sibling_indices": torch.tensor([0, 0], dtype=torch.long),
        "parent_key_ids": torch.tensor([1, 1], dtype=torch.long),
        "labels": torch.tensor([6, -100], dtype=torch.long),
        "kind_ids": torch.tensor([5, 5], dtype=torch.long),
    }

    batch = collate_fn([item1, item2])

    assert "kind_ids" in batch
    assert batch["kind_ids"].shape == (2, 3)
    assert batch["kind_ids"][1, 2].item() == 0  # padded with 0


def test_v4_dataset_skips_unk_targets():
    """Lever 1: positions whose target encodes to [UNK] should never be masked.

    Uses metadata.* keys (parent is a UNIVERSAL_ROOT_KEY so they route to
    the simple_head, not the kind_head — making the test independent of
    kind_target_vocab). Sets mask_prob=1.0 so every eligible key WOULD be
    masked under the old code. Verifies the [UNK]-target position keeps
    labels==-100 (skipped) while the known-target position is supervised.
    """
    import random
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.dataset import YamlDataset
    from yaml_bert.types import NodeType, YamlNode
    from yaml_bert.vocab import Vocabulary

    specials = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2}
    key_vocab = {"apiVersion": 3, "kind": 4, "metadata": 5, "name": 6, "unknownkey": 7}
    value_vocab = {"v1": 3, "Pod": 4, "app": 5, "noise": 6}
    # ONLY 'metadata::name' is a known target. 'metadata::unknownkey' is NOT.
    simple_target_vocab = {"apiVersion": 0, "kind": 1, "metadata": 2, "metadata::name": 3}
    vocab = Vocabulary(
        key_vocab=key_vocab, value_vocab=value_vocab, special_tokens=specials,
        simple_target_vocab=simple_target_vocab, kind_target_vocab={},
    )

    # apiVersion=v1, kind=Pod, metadata: {name: app, unknownkey: noise}
    doc = [
        YamlNode("apiVersion", NodeType.KEY,   depth=0, sibling_index=0, parent_path=""),
        YamlNode("v1",         NodeType.VALUE, depth=0, sibling_index=0, parent_path="apiVersion"),
        YamlNode("kind",       NodeType.KEY,   depth=0, sibling_index=1, parent_path=""),
        YamlNode("Pod",        NodeType.VALUE, depth=0, sibling_index=1, parent_path="kind"),
        YamlNode("metadata",   NodeType.KEY,   depth=0, sibling_index=2, parent_path=""),
        YamlNode("name",       NodeType.KEY,   depth=1, sibling_index=0, parent_path="metadata"),
        YamlNode("app",        NodeType.VALUE, depth=1, sibling_index=0, parent_path="metadata.name"),
        YamlNode("unknownkey", NodeType.KEY,   depth=1, sibling_index=1, parent_path="metadata"),
        YamlNode("noise",      NodeType.VALUE, depth=1, sibling_index=1, parent_path="metadata.unknownkey"),
    ]

    config = YamlBertConfig(mask_prob=1.0, skip_unk_targets=True)
    dataset = YamlDataset.from_cached_docs_v4([doc], vocab, config)

    unk_id = vocab.special_tokens["[UNK]"]
    name_pos = 5         # 'name' key — target 'metadata::name' (known)
    unknown_pos = 7      # 'unknownkey' — target 'metadata::unknownkey' ([UNK])

    for trial in range(20):
        random.seed(trial)
        item = dataset[0]
        # [UNK]-target position should never be supervised
        assert item["simple_labels"][unknown_pos].item() == -100, \
            f"trial {trial}: [UNK]-target position {unknown_pos} should not be supervised"
        assert item["kind_labels"][unknown_pos].item() == -100
        # Known-target position should be supervised every trial (mask_prob=1.0)
        s = item["simple_labels"][name_pos].item()
        assert s != -100 and s != unk_id, \
            f"trial {trial}: known-target position should be supervised; got label={s}"


def test_v4_dataset_skip_unk_can_be_disabled():
    """With skip_unk_targets=False, [UNK]-target positions DO get masked
    (legacy behavior preserved for ablation comparison)."""
    import random
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.dataset import YamlDataset
    from yaml_bert.types import NodeType, YamlNode
    from yaml_bert.vocab import Vocabulary

    specials = {"[PAD]": 0, "[UNK]": 1, "[MASK]": 2}
    key_vocab = {"metadata": 3, "unknownkey": 4}
    value_vocab = {"noise": 3}
    simple_target_vocab = {"metadata": 0}  # 'metadata::unknownkey' NOT here
    vocab = Vocabulary(
        key_vocab=key_vocab, value_vocab=value_vocab, special_tokens=specials,
        simple_target_vocab=simple_target_vocab, kind_target_vocab={},
    )
    doc = [
        YamlNode("metadata",   NodeType.KEY,   depth=0, sibling_index=0, parent_path=""),
        YamlNode("unknownkey", NodeType.KEY,   depth=1, sibling_index=0, parent_path="metadata"),
        YamlNode("noise",      NodeType.VALUE, depth=1, sibling_index=0, parent_path="metadata.unknownkey"),
    ]

    config = YamlBertConfig(mask_prob=1.0, skip_unk_targets=False)
    dataset = YamlDataset.from_cached_docs_v4([doc], vocab, config)

    unk_id = vocab.special_tokens["[UNK]"]
    saw_unk_target = False
    for trial in range(20):
        random.seed(trial)
        item = dataset[0]
        if item["simple_labels"][1].item() == unk_id:
            saw_unk_target = True
            break
    assert saw_unk_target, \
        "With skip_unk_targets=False, [UNK]-target positions should still be supervised"


def test_v4_dataset_hybrid_labels():
    vocab = _build_vocab()
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    import glob
    docs = []
    for path in sorted(glob.glob(os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True)):
        nodes = linearizer.linearize_file(path)
        if nodes:
            annotator.annotate(nodes)
            docs.append(nodes)

    dataset = YamlDataset.from_cached_docs_v4(docs, vocab)
    item = dataset[0]

    assert "simple_labels" in item
    assert "kind_labels" in item
    assert "parent_key_ids" not in item
    assert "kind_ids" not in item
    assert "labels" not in item

    # Check that masked positions have label in exactly one of simple/kind
    simple = item["simple_labels"]
    kind = item["kind_labels"]
    masked = (simple != -100) | (kind != -100)
    if masked.any():
        # No position should have both set
        both = (simple != -100) & (kind != -100)
        assert not both.any(), "A position has both simple and kind labels"
