"""Integration test: YamlBertDataset emits subtree-masking outputs; collate_fn batches them."""
import torch

from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.config import YamlBertConfig
from yaml_bert.vocab import VocabBuilder
from yaml_bert.dataset import YamlBertDataset, collate_fn


def _build_dataset_and_vocab(yamls: list[str], recon_enabled: bool):
    docs = [YamlLinearizer().linearize(y) for y in yamls]
    flat = [n for d in docs for n in d]
    vocab = VocabBuilder().build(flat, min_freq=1)
    config = YamlBertConfig(mask_prob=0.0,
                            recon_enabled=recon_enabled)
    return YamlBertDataset(docs, vocab, config), vocab


def test_dataset_item_includes_subtree_fields_when_recon_enabled():
    """When recon_enabled=True, items carry subtree_mask + subtree_roots
    + bag_of_keys_targets."""
    yamls = [
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: web\n"
        "spec:\n  replicas: 3\n  template:\n    spec:\n      containers:\n"
        "      - name: nginx\n        image: nginx:1.25\n        ports:\n"
        "        - containerPort: 80\n",
    ]
    ds, _ = _build_dataset_and_vocab(yamls, recon_enabled=True)
    item = ds[0]

    assert "subtree_mask" in item
    assert item["subtree_mask"].dtype == torch.bool
    assert item["subtree_mask"].shape[0] == item["token_ids"].shape[0]

    assert "subtree_roots" in item
    assert isinstance(item["subtree_roots"], list)
    assert all(isinstance(r, int) for r in item["subtree_roots"])

    assert "bag_of_keys_targets" in item
    # If subtree_roots is non-empty, each root has a multi-hot target of length V_atomic
    if item["subtree_roots"]:
        assert len(item["bag_of_keys_targets"]) == len(item["subtree_roots"])
        for target in item["bag_of_keys_targets"]:
            assert target.dtype == torch.float
            assert target.dim() == 1


def test_dataset_item_omits_subtree_fields_when_recon_disabled():
    """When recon_enabled=False, no subtree-related fields appear in item."""
    yamls = ["apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n"]
    ds, _ = _build_dataset_and_vocab(yamls, recon_enabled=False)
    item = ds[0]
    assert "subtree_mask" not in item
    assert "subtree_roots" not in item
    assert "bag_of_keys_targets" not in item


def test_collate_batches_subtree_fields_when_present():
    """collate_fn batches per-item subtree fields into (B,N) + flat (M,*).

    Structural assertions hold regardless of whether any subtrees were picked
    (pick_subtrees returns [] when no candidates pass the size cap).
    """
    yamls = [
        "apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: a\n"
        "spec:\n  template:\n    spec:\n      containers:\n      - name: x\n",
        "apiVersion: v1\nkind: Pod\nmetadata:\n  name: b\n"
        "spec:\n  containers:\n  - name: y\n    image: nginx\n",
    ]
    ds, vocab = _build_dataset_and_vocab(yamls, recon_enabled=True)
    batch = collate_fn([ds[0], ds[1]])

    assert "subtree_mask" in batch
    assert batch["subtree_mask"].dim() == 2  # (B, N)
    assert batch["subtree_mask"].shape[0] == 2
    assert batch["subtree_mask"].dtype == torch.bool

    assert "subtree_roots_flat" in batch
    sr = batch["subtree_roots_flat"]
    assert sr.dtype == torch.long
    assert sr.dim() == 2 and sr.shape[1] == 2  # (M, 2) of [batch_idx, root_pos]

    assert "bag_of_keys_targets_flat" in batch
    bot = batch["bag_of_keys_targets_flat"]
    assert bot.dtype == torch.float
    assert bot.dim() == 2  # (M, V_atomic)
    assert bot.shape[0] == sr.shape[0]
    # shape[1] == V_atomic regardless of whether picks are empty
    assert bot.shape[1] == vocab.atomic_target_vocab_size


def test_collate_omits_subtree_fields_when_absent():
    """If items don't carry subtree fields, neither does the batched dict."""
    yamls = ["apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n",
             "apiVersion: v1\nkind: Service\n"]
    ds, _ = _build_dataset_and_vocab(yamls, recon_enabled=False)
    batch = collate_fn([ds[0], ds[1]])
    assert "subtree_mask" not in batch
    assert "subtree_roots_flat" not in batch
    assert "bag_of_keys_targets_flat" not in batch
