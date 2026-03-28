from __future__ import annotations

import glob
import os
import random

import torch
from torch.utils.data import Dataset

from yaml_bert.annotator import DomainAnnotator
from yaml_bert.config import YamlBertConfig
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.types import NodeType, YamlNode
from yaml_bert.vocab import Vocabulary


# NodeType to integer index mapping
_NODE_TYPE_INDEX: dict[NodeType, int] = {
    NodeType.KEY: 0,
    NodeType.VALUE: 1,
    NodeType.LIST_KEY: 2,
    NodeType.LIST_VALUE: 3,
}

# Node types eligible for masking
_MASKABLE_TYPES: set[NodeType] = {NodeType.KEY, NodeType.LIST_KEY}


class YamlDataset(Dataset):
    """Dataset of linearized, masked YAML documents for YAML-BERT training."""

    def __init__(
        self,
        yaml_dir: str,
        vocab: Vocabulary,
        linearizer: YamlLinearizer,
        annotator: DomainAnnotator,
        config: YamlBertConfig | None = None,
    ) -> None:
        config = config or YamlBertConfig()
        self.vocab: Vocabulary = vocab
        self.linearizer: YamlLinearizer = linearizer
        self.annotator: DomainAnnotator = annotator
        self.mask_prob: float = config.mask_prob
        self.max_seq_len: int = config.max_seq_len

        # Load and linearize all YAML files
        yaml_files: list[str] = sorted(
            glob.glob(os.path.join(yaml_dir, "**", "*.yaml"), recursive=True)
        )
        self.documents: list[list[YamlNode]] = []
        for path in yaml_files:
            nodes: list[YamlNode] = linearizer.linearize_file(path)
            if nodes:
                annotator.annotate(nodes)
                self.documents.append(nodes)

    @classmethod
    def from_huggingface(
        cls,
        dataset_name: str,
        vocab: Vocabulary,
        linearizer: YamlLinearizer,
        annotator: DomainAnnotator,
        config: YamlBertConfig | None = None,
        max_docs: int | None = None,
    ) -> "YamlDataset":
        """Load dataset from HuggingFace Hub.

        Args:
            dataset_name: HuggingFace dataset ID, e.g. "substratusai/the-stack-yaml-k8s"
            vocab: Pre-built vocabulary
            linearizer: YamlLinearizer instance
            annotator: DomainAnnotator instance
            config: Optional config (uses defaults if None)
            max_docs: Load at most this many documents (None = all)
        """
        from datasets import load_dataset

        config = config or YamlBertConfig()
        instance: YamlDataset = cls.__new__(cls)
        instance.vocab = vocab
        instance.linearizer = linearizer
        instance.annotator = annotator
        instance.mask_prob = config.mask_prob
        instance.max_seq_len = config.max_seq_len
        instance.documents = []

        print(f"Loading dataset: {dataset_name}")
        ds = load_dataset(dataset_name, split="train")

        total: int = len(ds) if max_docs is None else min(max_docs, len(ds))
        print(f"Linearizing {total:,} / {len(ds):,} documents...")

        skipped: int = 0
        for i in range(total):
            yaml_content: str = ds[i]["content"]
            try:
                nodes: list[YamlNode] = linearizer.linearize(yaml_content)
            except Exception:
                skipped += 1
                continue
            if nodes:
                annotator.annotate(nodes)
                instance.documents.append(nodes)

            if (i + 1) % 10000 == 0:
                print(f"  {i + 1:,} / {total:,} processed ({skipped} skipped)")

        print(f"Loaded {len(instance.documents):,} documents ({skipped} skipped)")
        return instance

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        nodes: list[YamlNode] = self.documents[idx]

        # Truncate if needed
        if len(nodes) > self.max_seq_len:
            nodes = nodes[: self.max_seq_len]

        seq_len: int = len(nodes)

        # Encode nodes to integer arrays
        token_ids: list[int] = []
        node_types: list[int] = []
        depths: list[int] = []
        sibling_indices: list[int] = []
        parent_key_ids: list[int] = []

        for node in nodes:
            # Token ID — route to correct vocab
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                token_ids.append(self.vocab.encode_key(node.token))
            else:
                token_ids.append(self.vocab.encode_value(node.token))

            node_types.append(_NODE_TYPE_INDEX[node.node_type])
            depths.append(min(node.depth, 15))  # clamp to max_depth - 1
            sibling_indices.append(min(node.sibling_index, 31))  # clamp to max_sibling - 1

            parent_key: str = Vocabulary.extract_parent_key(node.parent_path)
            parent_key_ids.append(self.vocab.encode_key(parent_key))

        # Apply masking (only to KEY and LIST_KEY nodes)
        labels: list[int] = [-100] * seq_len
        mask_token_id: int = self.vocab.special_tokens["[MASK]"]

        for i in range(seq_len):
            if nodes[i].node_type not in _MASKABLE_TYPES:
                continue
            if random.random() >= self.mask_prob:
                continue

            # This position is selected for masking
            labels[i] = token_ids[i]  # save original token ID as label

            rand: float = random.random()
            if rand < 0.8:
                # 80%: replace with [MASK]
                token_ids[i] = mask_token_id
            elif rand < 0.9:
                # 10%: replace with random key
                random_key_id: int = random.randint(
                    len(self.vocab.special_tokens),
                    len(self.vocab.key_vocab) + len(self.vocab.special_tokens) - 1,
                )
                token_ids[i] = random_key_id
            # else 10%: keep unchanged

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "node_types": torch.tensor(node_types, dtype=torch.long),
            "depths": torch.tensor(depths, dtype=torch.long),
            "sibling_indices": torch.tensor(sibling_indices, dtype=torch.long),
            "parent_key_ids": torch.tensor(parent_key_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Pad a batch of variable-length sequences and create padding mask."""
    max_len: int = max(item["token_ids"].size(0) for item in batch)

    padded: dict[str, list[torch.Tensor]] = {key: [] for key in batch[0].keys()}
    padding_masks: list[torch.Tensor] = []

    for item in batch:
        seq_len: int = item["token_ids"].size(0)
        pad_len: int = max_len - seq_len

        for key in item:
            if pad_len > 0:
                pad_value: int = -100 if key == "labels" else 0
                padding: torch.Tensor = torch.full(
                    (pad_len,), pad_value, dtype=torch.long
                )
                padded[key].append(torch.cat([item[key], padding]))
            else:
                padded[key].append(item[key])

        mask: torch.Tensor = torch.cat([
            torch.zeros(seq_len, dtype=torch.bool),
            torch.ones(pad_len, dtype=torch.bool),
        ]) if pad_len > 0 else torch.zeros(seq_len, dtype=torch.bool)
        padding_masks.append(mask)

    result: dict[str, torch.Tensor] = {
        key: torch.stack(tensors) for key, tensors in padded.items()
    }
    result["padding_mask"] = torch.stack(padding_masks)

    return result
