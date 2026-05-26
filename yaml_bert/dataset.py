"""v9 YAML-BERT dataset: BPE-expand each linearizer node into subword
positions, mask whole logical KEYs, emit per-logical-node atomic labels."""
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
    return _LIST_INDEX_RE.sub("", path)


def compute_children_info(nodes: list[YamlNode]) -> dict:
    """Same as v8 — operates on LOGICAL nodes (not subwords)."""
    n = len(nodes)
    full_path_of: list[str] = []
    for node in nodes:
        if node.parent_path:
            full_path_of.append(f"{node.parent_path}.{node.token}")
        else:
            full_path_of.append(node.token)

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
        parent_pos = path_to_key_pos.get(parent_path)
        if parent_pos is None:
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


class YamlBertDataset(Dataset):
    """v9 dataset: subword expansion + whole-key MLM masking + recon."""

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

        self._cached_children_info: list[dict] = []
        self._cached_descendants: list[dict[int, set[int]] | None] = []
        for doc in documents:
            # Cap LOGICAL nodes here; BPE expansion may still exceed max_seq_len
            # at the subword level (handled below by truncation).
            ci = compute_children_info(doc)
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

        # Pass 1: BPE-expand each logical node, building per-subword tensors.
        sub_token_ids: list[int] = []
        sub_node_types: list[int] = []
        sub_depths: list[int] = []
        sub_sibling: list[int] = []
        sub_logical_ids: list[int] = []
        per_logical_subword_spans: list[tuple[int, int]] = []
        # We may need to drop trailing logical nodes if subword expansion
        # blows the max_seq_len cap.
        kept_logical: int = 0
        for logical_idx, node in enumerate(nodes):
            is_value = node.node_type in (NodeType.VALUE, NodeType.LIST_VALUE)
            ids = self.vocab.encode_token(node.token, is_value=is_value)
            if len(sub_token_ids) + len(ids) > self.max_seq_len:
                break
            start = len(sub_token_ids)
            sub_token_ids.extend(ids)
            sub_node_types.extend([_NODE_TYPE_INDEX[node.node_type]] * len(ids))
            sub_depths.extend([min(node.depth, 15)] * len(ids))
            sub_sibling.extend([min(node.sibling_index, 31)] * len(ids))
            sub_logical_ids.extend([kept_logical] * len(ids))
            per_logical_subword_spans.append((start, start + len(ids)))
            kept_logical += 1

        n_logical = kept_logical
        # Truncate cached children_info to kept logicals
        ci = self._cached_children_info[idx]
        kept_set = set(range(n_logical))
        children_of_t = [
            [c for c in ci["children_of"][p] if c in kept_set]
            for p in range(n_logical)
        ]
        parent_of_t = [
            ci["parent_of"][p] if (ci["parent_of"][p] in kept_set or ci["parent_of"][p] == -1) else -1
            for p in range(n_logical)
        ]
        key_positions_t = [p for p in ci["key_positions"] if p < n_logical]
        depth_of_t = ci["depth_of"][:n_logical]
        full_path_of_t = ci["full_path_of"][:n_logical]
        ci_t = {
            "children_of": children_of_t,
            "parent_of": parent_of_t,
            "key_positions": key_positions_t,
            "depth_of": depth_of_t,
            "full_path_of": full_path_of_t,
        }

        # Pass 2: whole-key MLM masking, one decision per LOGICAL KEY.
        atomic_labels: list[int] = [-100] * n_logical
        mask_id = self.vocab.mask_id
        unk_id = self.vocab.unk_id
        subword_vocab_size = self.vocab.subword_vocab_size

        mlm_masked_positions: set[int] = set()
        for logical_idx in key_positions_t:
            if random.random() >= self.mask_prob:
                continue
            tok = nodes[logical_idx].token
            atomic_id = self.vocab.encode_atomic_target(tok)
            if atomic_id == unk_id:
                continue  # skip [UNK] targets (Lever 1, carried from v8)
            atomic_labels[logical_idx] = atomic_id
            mlm_masked_positions.add(logical_idx)
            r = random.random()
            start, end = per_logical_subword_spans[logical_idx]
            if r < 0.8:
                for p in range(start, end):
                    sub_token_ids[p] = mask_id
            elif r < 0.9:
                for p in range(start, end):
                    sub_token_ids[p] = random.randint(4, subword_vocab_size - 1)

        result = {
            "token_ids": torch.tensor(sub_token_ids, dtype=torch.long),
            "node_types": torch.tensor(sub_node_types, dtype=torch.long),
            "depths": torch.tensor(sub_depths, dtype=torch.long),
            "sibling_indices": torch.tensor(sub_sibling, dtype=torch.long),
            "logical_ids": torch.tensor(sub_logical_ids, dtype=torch.long),
            "atomic_labels": torch.tensor(atomic_labels, dtype=torch.long),
            "children_info": ci_t,
            "n_logical": n_logical,
        }

        if self.recon_enabled:
            from yaml_bert.subtree_masking import pick_subtrees, bag_of_keys_target
            picked_roots = pick_subtrees(
                N=n_logical,
                key_positions=key_positions_t,
                depth_of=depth_of_t,
                children_of=children_of_t,
                mlm_masked_positions=mlm_masked_positions,
                rng=random,
                descendants_cache={
                    kp: descendants_of(kp, children_of_t)
                    for kp in key_positions_t
                    if children_of_t[kp]
                },
            )
            subtree_mask = torch.zeros(n_logical, dtype=torch.bool)
            picked_positions_all: set[int] = set()
            bag_targets: list[torch.Tensor] = []
            position_to_key_str = {
                i: nodes[i].token for i in key_positions_t
            }
            for root_pos in picked_roots:
                descs = {
                    d for d in descendants_of(root_pos, children_of_t)
                    if d < n_logical
                }
                picked_positions_all |= descs
                bag_targets.append(bag_of_keys_target(
                    subtree_positions=descs,
                    position_to_key_str=position_to_key_str,
                    atomic_vocab=self.vocab.atomic_target_vocab,
                    vocab_size=self.vocab.atomic_target_vocab_size,
                ))
            # Apply [MASK] to ALL subwords of each logical position in the picked subtree
            for lpos in picked_positions_all:
                subtree_mask[lpos] = True
                start, end = per_logical_subword_spans[lpos]
                for p in range(start, end):
                    sub_token_ids[p] = mask_id
            result["token_ids"] = torch.tensor(sub_token_ids, dtype=torch.long)
            result["subtree_mask"] = subtree_mask
            result["subtree_roots"] = picked_roots
            result["bag_of_keys_targets"] = bag_targets
            result["_atomic_vocab_size"] = self.vocab.atomic_target_vocab_size

        return result


_COLLATE_NON_TENSOR_KEYS = frozenset({
    "children_info",
    "subtree_roots",
    "bag_of_keys_targets",
    "subtree_mask",
    "_atomic_vocab_size",
    "n_logical",
})


def collate_fn(batch: list[dict]) -> dict:
    """Pad subword-level tensors AND logical-level tensors.

    Subword-level (per-position): token_ids, node_types, depths, sibling_indices,
                                  logical_ids  — padded to max subword length
    Logical-level (per-logical-node): atomic_labels  — padded to max logical count
    """
    max_sub_len = max(item["token_ids"].size(0) for item in batch)
    max_logical = max(item["n_logical"] for item in batch)

    subword_keys = ("token_ids", "node_types", "depths", "sibling_indices",
                    "logical_ids")
    padded_sub: dict[str, list[torch.Tensor]] = {k: [] for k in subword_keys}
    padded_labels: list[torch.Tensor] = []
    padding_masks: list[torch.Tensor] = []
    batch_info: list[dict] = []
    n_logical_per_doc: list[int] = []

    for item in batch:
        sub_len = item["token_ids"].size(0)
        pad_sub = max_sub_len - sub_len
        for k in subword_keys:
            pad_value = -1 if k == "logical_ids" else 0
            if pad_sub > 0:
                padding = torch.full((pad_sub,), pad_value, dtype=torch.long)
                padded_sub[k].append(torch.cat([item[k], padding]))
            else:
                padded_sub[k].append(item[k])

        labels = item["atomic_labels"]
        pad_lab = max_logical - labels.size(0)
        if pad_lab > 0:
            padded_labels.append(torch.cat([
                labels, torch.full((pad_lab,), -100, dtype=torch.long),
            ]))
        else:
            padded_labels.append(labels)

        mask = torch.cat([
            torch.zeros(sub_len, dtype=torch.bool),
            torch.ones(pad_sub, dtype=torch.bool),
        ]) if pad_sub > 0 else torch.zeros(sub_len, dtype=torch.bool)
        padding_masks.append(mask)
        batch_info.append(item["children_info"])
        n_logical_per_doc.append(item["n_logical"])

    result = {k: torch.stack(v) for k, v in padded_sub.items()}
    result["atomic_labels"] = torch.stack(padded_labels)
    result["padding_mask"] = torch.stack(padding_masks)
    result["batch_info"] = batch_info
    result["n_logical_per_doc"] = torch.tensor(n_logical_per_doc, dtype=torch.long)

    # parent_of_tensor and top_level_key_mask now operate at LOGICAL level
    B = len(batch)
    L = max_logical
    parent_of_tensor = torch.full((B, L), -1, dtype=torch.long)
    top_level_key_mask = torch.zeros((B, L), dtype=torch.bool)
    for b_idx, info in enumerate(batch_info):
        parent_of = info["parent_of"]
        n_b = len(parent_of)
        if n_b > 0:
            parent_of_tensor[b_idx, :n_b] = torch.tensor(parent_of, dtype=torch.long)
        depth_of = info["depth_of"]
        depth_zero_kps = [kp for kp in info["key_positions"] if depth_of[kp] == 0]
        if depth_zero_kps:
            top_level_key_mask[b_idx, depth_zero_kps] = True

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

    if "subtree_mask" in batch[0]:
        subtree_masks: list[torch.Tensor] = []
        for item in batch:
            sm = item["subtree_mask"]
            pad_len = max_logical - sm.size(0)
            if pad_len > 0:
                subtree_masks.append(torch.cat([
                    sm, torch.zeros(pad_len, dtype=torch.bool),
                ]))
            else:
                subtree_masks.append(sm)
        result["subtree_mask"] = torch.stack(subtree_masks)

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
            result["subtree_roots_flat"] = torch.zeros((0, 2), dtype=torch.long)
            v = batch[0].get("_atomic_vocab_size", 0)
            result["bag_of_keys_targets_flat"] = torch.zeros(
                (0, v), dtype=torch.float,
            )

    return result
