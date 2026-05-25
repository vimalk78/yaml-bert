"""Subtree-masking primitives for the v8 reconstruction objective.

Pure functions (no torch in the masking logic itself; targets returned as
torch tensors). All randomness flows through a caller-provided random.Random
instance for reproducibility.

Notes on children_of typing:
    compute_children_info() returns children_of as list[list[int]] (indexed by
    position). The functions here accept any Mapping or Sequence via a unified
    helper _get_children(), so they work with both dict (unit tests) and list
    (dataset call sites).
"""
from __future__ import annotations

import random
from typing import Mapping, Sequence, Union

import torch


MIN_DOC_NODES = 10
MAX_SUBTREE_FRACTION = 0.30  # no single subtree may exceed this fraction of doc
MAX_TOTAL_SUBTREE_FRACTION = 0.05  # sum of all picked subtrees ≤ this fraction

# Accept both dict-style (unit tests) and list-style (dataset) children_of.
_ChildrenOf = Union[Mapping[int, Sequence[int]], Sequence[Sequence[int]]]


def _get_children(pos: int, children_of: _ChildrenOf) -> Sequence[int]:
    """Return children of pos, working for both dict and list children_of."""
    if isinstance(children_of, Mapping):
        return children_of.get(pos, [])  # type: ignore[return-value]
    # list[list[int]] — index directly; return [] if pos out of range
    if pos < len(children_of):  # type: ignore[arg-type]
        return children_of[pos]
    return []


def descendants_of(pos: int, children_of: _ChildrenOf) -> set[int]:
    """DFS descendant set of `pos`, including pos itself."""
    out = {pos}
    stack = list(_get_children(pos, children_of))
    while stack:
        p = stack.pop()
        out.add(p)
        stack.extend(_get_children(p, children_of))
    return out


def pick_subtrees(
    N: int,
    key_positions: Sequence[int],
    depth_of: Union[Mapping[int, int], Sequence[int]],
    children_of: _ChildrenOf,
    mlm_masked_positions: set[int],
    rng: random.Random,
    descendants_cache: Mapping[int, set[int]] | None = None,
) -> list[int]:
    """Pick 1-3 mutually-disjoint subtree root positions (or [] if none qualify).

    Args:
        N: total node count in the doc.
        key_positions: positions that are KEY/LIST_KEY.
        depth_of: per-position depth lookup (dict or list).
        children_of: per-position children list (dict or list).
        mlm_masked_positions: positions already masked by token-level MLM.
        rng: random source.
        descendants_cache: optional precomputed descendants per position
            (avoids redoing DFS on every call). If None, computed on the fly.

    Returns:
        List of 1-3 root positions, mutually-disjoint, total size within cap,
        or [] if doc too small / no valid candidates / cap too tight.
    """
    if N < MIN_DOC_NODES:
        return []

    def _descs(p: int) -> set[int]:
        if descendants_cache is not None and p in descendants_cache:
            return descendants_cache[p]
        return descendants_of(p, children_of)

    candidates: list[tuple[int, set[int]]] = []
    for kp in key_positions:
        if depth_of[kp] < 1:
            continue
        if not _get_children(kp, children_of):
            continue
        descs = _descs(kp)
        if len(descs) > MAX_SUBTREE_FRACTION * N:
            continue
        if descs & mlm_masked_positions:
            continue
        candidates.append((kp, descs))

    if not candidates:
        return []

    rng.shuffle(candidates)
    num_to_pick = rng.randint(1, min(3, len(candidates)))

    picked: list[int] = []
    picked_positions: set[int] = set()
    for kp, descs in candidates:
        if descs & picked_positions:
            continue
        if len(picked_positions | descs) > MAX_TOTAL_SUBTREE_FRACTION * N:
            continue
        picked.append(kp)
        picked_positions.update(descs)
        if len(picked) >= num_to_pick:
            break
    return picked


def bag_of_keys_target(
    subtree_positions: set[int],
    position_to_key_str: Mapping[int, str],
    atomic_vocab: Mapping[str, int],
    vocab_size: int,
) -> torch.Tensor:
    """Build a multi-hot bag-of-keys target for one masked subtree.

    Args:
        subtree_positions: all positions inside the subtree (root + descendants).
        position_to_key_str: per-position key string lookup (positions not in this
            map are skipped — e.g., VALUE positions don't contribute key tokens).
        atomic_vocab: atomic key string → vocab index.
        vocab_size: V_atomic, size of the output multi-hot vector.

    Returns:
        Float tensor of shape (vocab_size,). target[v]=1.0 iff vocab index v
        appears as a key inside the subtree. De-duplicated (true bag-of-keys,
        not bag-of-counts).
    """
    target = torch.zeros(vocab_size, dtype=torch.float)
    seen: set[int] = set()
    for pos in subtree_positions:
        key_str = position_to_key_str.get(pos)
        if key_str is None:
            continue
        vocab_idx = atomic_vocab.get(key_str)
        if vocab_idx is None:
            continue
        if vocab_idx in seen:
            continue
        target[vocab_idx] = 1.0
        seen.add(vocab_idx)
    return target
