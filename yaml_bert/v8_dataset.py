"""v8 dataset extensions: children-info precompute for tree aggregator."""
from __future__ import annotations

import random
import re

import torch
from torch.utils.data import Dataset

from yaml_bert.config import YamlBertConfig
from yaml_bert.subtree_masking import descendants_of
from yaml_bert.types import NodeType, YamlNode
from yaml_bert.vocab import Vocabulary


_LIST_INDEX_RE = re.compile(r"\.\d+$")


def _strip_trailing_list_index(path: str) -> str:
    """Strip a trailing numeric segment from a parent_path.

    E.g., 'spec.containers.0' -> 'spec.containers'.
    Returns the path unchanged if no trailing numeric segment exists.
    """
    return _LIST_INDEX_RE.sub("", path)


def compute_children_info(nodes: list[YamlNode]) -> dict:
    """Compute per-position parent/child relationships.

    For each KEY/LIST_KEY position, its children are KEY/LIST_KEY positions
    whose parent_path equals this key's full_path. (VALUE nodes are leaves —
    their hidden states are used in the aggregator without being treated as
    "children" of any KEY for subtree purposes.)

    For KEYs inside list items, parent_path ends in a numeric list index
    (e.g., 'spec.containers.0'), but the linearizer never emits a synthetic
    list-item node, so a direct lookup misses. We strip the trailing numeric
    segment and link to the list-key itself. This flattens all list items
    into their list parent — per-item grouping is lost, but the aggregator
    still produces a meaningful doc vector. Phase 1 may add synthetic
    list-item nodes if per-item grouping matters.

    Returns a dict with:
        children_of: list[list[int]] — children's positions per node
        parent_of:   list[int]       — parent position (or -1)
        key_positions: list[int]     — positions that are KEY/LIST_KEY
        depth_of:    list[int]       — depth per position
        full_path_of: list[str]      — full path per position
    """
    n = len(nodes)
    full_path_of: list[str] = []
    for node in nodes:
        if node.parent_path:
            full_path_of.append(f"{node.parent_path}.{node.token}")
        else:
            full_path_of.append(node.token)

    # Index KEY positions by their full_path
    key_positions: list[int] = [
        i for i, node in enumerate(nodes)
        if node.node_type in (NodeType.KEY, NodeType.LIST_KEY)
    ]
    path_to_key_pos: dict[str, int] = {
        full_path_of[p]: p for p in key_positions
    }

    children_of: list[list[int]] = [[] for _ in range(n)]
    parent_of: list[int] = [-1] * n
    depth_of: list[int] = [node.depth for node in nodes]

    for p in key_positions:
        parent_path = nodes[p].parent_path
        if not parent_path:
            continue
        # Try direct lookup first
        parent_pos = path_to_key_pos.get(parent_path)
        if parent_pos is None:
            # Try with trailing list index stripped
            # (e.g., "spec.containers.0" -> "spec.containers")
            stripped = _strip_trailing_list_index(parent_path)
            if stripped != parent_path:
                parent_pos = path_to_key_pos.get(stripped)
        if parent_pos is not None:
            parent_of[p] = parent_pos
            children_of[parent_pos].append(p)

    return {
        "children_of": children_of,
        "parent_of": parent_of,
        "key_positions": key_positions,
        "depth_of": depth_of,
        "full_path_of": full_path_of,
    }


_NODE_TYPE_INDEX = {
    NodeType.KEY: 0,
    NodeType.VALUE: 1,
    NodeType.LIST_KEY: 2,
    NodeType.LIST_VALUE: 3,
}
_MASKABLE_TYPES = (NodeType.KEY, NodeType.LIST_KEY)


class V8Dataset(Dataset):
    """v8 dataset: atomic labels at masked positions + children info per doc."""

    def __init__(
        self,
        documents: list[list[YamlNode]],
        vocab: Vocabulary,
        config: YamlBertConfig,
    ) -> None:
        self.documents = documents
        self.vocab = vocab
        self.mask_prob = config.mask_prob
        self.max_seq_len = config.max_seq_len
        self.recon_enabled = config.recon_enabled

        # Precompute per-doc children_info AND descendant cache (subtree picker
        # uses descendants on every __getitem__ call; cache once at init).
        self._cached_children_info: list[dict] = []
        self._cached_descendants: list[dict[int, set[int]] | None] = []
        if self.recon_enabled and len(documents) > 100:
            import logging
            logging.getLogger(__name__).info(
                "V8Dataset: precomputing descendants for %d docs (recon enabled)",
                len(documents),
            )
        for doc in documents:
            ci = compute_children_info(doc[: self.max_seq_len])
            self._cached_children_info.append(ci)
            if self.recon_enabled:
                desc_cache: dict[int, set[int]] = {}
                for kp in ci["key_positions"]:
                    if ci["children_of"][kp]:
                        desc_cache[kp] = descendants_of(kp, ci["children_of"])
                self._cached_descendants.append(desc_cache)
            else:
                self._cached_descendants.append(None)

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, idx: int) -> dict:
        nodes = self.documents[idx]
        if len(nodes) > self.max_seq_len:
            nodes = nodes[: self.max_seq_len]

        token_ids: list[int] = []
        node_types: list[int] = []
        depths: list[int] = []
        sibling_indices: list[int] = []

        for node in nodes:
            if node.node_type in (NodeType.KEY, NodeType.LIST_KEY):
                token_ids.append(self.vocab.encode_key(node.token))
            else:
                token_ids.append(self.vocab.encode_value(node.token))
            node_types.append(_NODE_TYPE_INDEX[node.node_type])
            depths.append(min(node.depth, 15))
            sibling_indices.append(min(node.sibling_index, 31))

        atomic_labels: list[int] = [-100] * len(nodes)
        mask_id: int = self.vocab.special_tokens["[MASK]"]
        unk_id: int = self.vocab.special_tokens["[UNK]"]

        mlm_masked_positions: set[int] = set()
        for i, node in enumerate(nodes):
            if node.node_type not in _MASKABLE_TYPES:
                continue
            if random.random() >= self.mask_prob:
                continue
            atomic_id = self.vocab.encode_atomic_target(node.token)
            if atomic_id == unk_id:
                continue  # skip [UNK] targets (Lever 1)
            atomic_labels[i] = atomic_id
            mlm_masked_positions.add(i)
            r = random.random()
            if r < 0.8:
                token_ids[i] = mask_id
            elif r < 0.9:
                token_ids[i] = random.randint(
                    len(self.vocab.special_tokens),
                    len(self.vocab.key_vocab) + len(self.vocab.special_tokens) - 1,
                )

        result = {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "node_types": torch.tensor(node_types, dtype=torch.long),
            "depths": torch.tensor(depths, dtype=torch.long),
            "sibling_indices": torch.tensor(sibling_indices, dtype=torch.long),
            "atomic_labels": torch.tensor(atomic_labels, dtype=torch.long),
            "children_info": self._cached_children_info[idx],
        }

        if self.recon_enabled:
            from yaml_bert.subtree_masking import pick_subtrees, bag_of_keys_target
            ci = self._cached_children_info[idx]
            picked_roots = pick_subtrees(
                N=len(nodes),
                key_positions=ci["key_positions"],
                depth_of=ci["depth_of"],
                children_of=ci["children_of"],
                mlm_masked_positions=mlm_masked_positions,
                rng=random,
                descendants_cache=self._cached_descendants[idx],
            )
            # Build subtree_mask + apply [MASK] to subtree positions
            subtree_mask = torch.zeros(len(nodes), dtype=torch.bool)
            picked_positions_all: set[int] = set()
            bag_targets: list[torch.Tensor] = []
            # position → key string lookup for bag-of-keys building
            position_to_key_str = {
                i: nodes[i].token
                for i in range(len(nodes))
                if nodes[i].node_type in (NodeType.KEY, NodeType.LIST_KEY)
            }
            for root_pos in picked_roots:
                descs = self._cached_descendants[idx][root_pos]
                picked_positions_all |= descs
                bag_targets.append(bag_of_keys_target(
                    subtree_positions=descs,
                    position_to_key_str=position_to_key_str,
                    atomic_vocab=self.vocab.atomic_target_vocab,
                    vocab_size=self.vocab.atomic_target_vocab_size,
                ))
            for pos in picked_positions_all:
                subtree_mask[pos] = True
                token_ids[pos] = mask_id
            # Re-encode token_ids tensor since we mutated the list
            result["token_ids"] = torch.tensor(token_ids, dtype=torch.long)
            result["subtree_mask"] = subtree_mask
            result["subtree_roots"] = picked_roots  # list[int]
            result["bag_of_keys_targets"] = bag_targets  # list[Tensor (V,)]
            result["_atomic_vocab_size"] = self.vocab.atomic_target_vocab_size

        return result


_COLLATE_NON_TENSOR_KEYS = frozenset({
    "children_info",
    "subtree_roots",
    "bag_of_keys_targets",
    "subtree_mask",  # bool tensor handled separately
    "_atomic_vocab_size",  # int, used by collate to size empty fallback tensors
})


def v8_collate_fn(batch: list[dict]) -> dict:
    """Pad tensor fields, keep children_info as a list."""
    max_len = max(item["token_ids"].size(0) for item in batch)
    padded: dict[str, list[torch.Tensor]] = {
        k: [] for k in batch[0].keys() if k not in _COLLATE_NON_TENSOR_KEYS
    }
    padding_masks: list[torch.Tensor] = []
    batch_info: list[dict] = []

    for item in batch:
        seq_len = item["token_ids"].size(0)
        pad_len = max_len - seq_len
        for key in padded:
            pad_value = -100 if key == "atomic_labels" else 0
            if pad_len > 0:
                padding = torch.full((pad_len,), pad_value, dtype=torch.long)
                padded[key].append(torch.cat([item[key], padding]))
            else:
                padded[key].append(item[key])
        mask = torch.cat([
            torch.zeros(seq_len, dtype=torch.bool),
            torch.ones(pad_len, dtype=torch.bool),
        ]) if pad_len > 0 else torch.zeros(seq_len, dtype=torch.bool)
        padding_masks.append(mask)
        batch_info.append(item["children_info"])

    result = {k: torch.stack(v) for k, v in padded.items()}
    result["padding_mask"] = torch.stack(padding_masks)
    result["batch_info"] = batch_info

    # Vectorized-aggregator precompute. CPU work, runs in DataLoader workers.
    B = len(batch)
    N = max_len

    # parent_of_tensor: (B, N) long. -1 sentinel for no-parent, non-key, or padding.
    parent_of_tensor = torch.full((B, N), -1, dtype=torch.long)
    for b_idx, info in enumerate(batch_info):
        parent_of = info["parent_of"]  # list[int] of length n_b
        n_b = len(parent_of)
        if n_b > 0:
            parent_of_tensor[b_idx, :n_b] = torch.tensor(parent_of, dtype=torch.long)

    # top_level_key_mask: (B, N) bool. True where depth==0 AND position is a KEY.
    top_level_key_mask = torch.zeros((B, N), dtype=torch.bool)
    for b_idx, info in enumerate(batch_info):
        depth_of = info["depth_of"]
        depth_zero_kps = [kp for kp in info["key_positions"] if depth_of[kp] == 0]
        if depth_zero_kps:
            top_level_key_mask[b_idx, depth_zero_kps] = True

    # edges_by_depth: dict[depth, (E, 3) long] of [doc_idx, child_pos, parent_pos] across batch.
    # parents_by_depth: dict[depth, (P, 2) long] of unique [doc_idx, parent_pos] with at-least-one-child.
    edges_by_depth: dict[int, list[tuple[int, int, int]]] = {}
    parents_set_by_depth: dict[int, set[tuple[int, int]]] = {}
    for b_idx, info in enumerate(batch_info):
        children_of = info["children_of"]
        depth_of = info["depth_of"]
        for parent_pos in info["key_positions"]:
            kids = children_of[parent_pos]
            if not kids:
                continue
            parent_depth = depth_of[parent_pos]
            edges_by_depth.setdefault(parent_depth, []).extend(
                (b_idx, child_pos, parent_pos) for child_pos in kids
            )
            parents_set_by_depth.setdefault(parent_depth, set()).add(
                (b_idx, parent_pos),
            )

    result["parent_of_tensor"] = parent_of_tensor
    result["top_level_key_mask"] = top_level_key_mask
    result["edges_by_depth"] = {
        d: torch.tensor(edges, dtype=torch.long)
        for d, edges in edges_by_depth.items()
    }
    result["parents_by_depth"] = {
        d: torch.tensor(sorted(parents_set), dtype=torch.long)
        for d, parents_set in parents_set_by_depth.items()
    }

    # Subtree-mask batching (only present when recon is enabled per-item).
    if "subtree_mask" in batch[0]:
        # subtree_mask: (B, N) bool, pad with False
        subtree_masks: list[torch.Tensor] = []
        for item in batch:
            sm = item["subtree_mask"]
            pad_len = max_len - sm.size(0)
            if pad_len > 0:
                subtree_masks.append(torch.cat([
                    sm, torch.zeros(pad_len, dtype=torch.bool),
                ]))
            else:
                subtree_masks.append(sm)
        result["subtree_mask"] = torch.stack(subtree_masks)

        # subtree_roots_flat: (M, 2) of [batch_idx, root_pos]; M = total roots
        # bag_of_keys_targets_flat: (M, V_atomic)
        flat_roots: list[tuple[int, int]] = []
        flat_targets: list[torch.Tensor] = []
        for b_idx, item in enumerate(batch):
            for root_pos, target in zip(
                item["subtree_roots"], item["bag_of_keys_targets"]
            ):
                flat_roots.append((b_idx, root_pos))
                flat_targets.append(target)
        if flat_roots:
            result["subtree_roots_flat"] = torch.tensor(
                flat_roots, dtype=torch.long,
            )
            result["bag_of_keys_targets_flat"] = torch.stack(flat_targets)
        else:
            # Even with recon enabled, all docs in this batch may have empty
            # picks (small docs, no candidates). Emit empty tensors so the
            # consumer can detect "no subtrees this batch."
            result["subtree_roots_flat"] = torch.zeros(
                (0, 2), dtype=torch.long,
            )
            v = batch[0].get("_atomic_vocab_size", 0)
            result["bag_of_keys_targets_flat"] = torch.zeros(
                (0, v), dtype=torch.float,
            )

    return result
