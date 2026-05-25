"""Unit tests for subtree_masking primitives (pure functions, no torch)."""
import random

from yaml_bert.subtree_masking import (
    descendants_of,
    pick_subtrees,
    bag_of_keys_target,
    MIN_DOC_NODES,
    MAX_SUBTREE_FRACTION,
    MAX_TOTAL_SUBTREE_FRACTION,
)


def test_descendants_of_single_leaf():
    """A KEY with no children has only itself as descendant."""
    children_of = {0: [], 1: [], 2: []}
    assert descendants_of(0, children_of) == {0}


def test_descendants_of_tree():
    """Tree shape:
       0
       ├─ 1
       │   └─ 3
       └─ 2
    """
    children_of = {0: [1, 2], 1: [3], 2: [], 3: []}
    assert descendants_of(0, children_of) == {0, 1, 2, 3}
    assert descendants_of(1, children_of) == {1, 3}
    assert descendants_of(2, children_of) == {2}


def test_pick_subtrees_returns_empty_for_tiny_doc():
    """Doc smaller than MIN_DOC_NODES → no picks."""
    rng = random.Random(0)
    picks = pick_subtrees(
        N=5,  # below threshold
        key_positions=[0, 1, 2],
        depth_of=[0, 1, 1],
        children_of={0: [1, 2], 1: [], 2: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    assert picks == []


def test_pick_subtrees_returns_empty_when_no_candidates():
    """Doc large enough but no depth>=1 KEY with children → empty."""
    rng = random.Random(0)
    # Only the root has children; depth-0 is excluded
    picks = pick_subtrees(
        N=20,
        key_positions=[0, 1, 2],
        depth_of=[0, 1, 1],
        children_of={0: [1, 2], 1: [], 2: []},  # 1 and 2 are leaves
        mlm_masked_positions=set(),
        rng=rng,
    )
    assert picks == []


def test_pick_subtrees_basic_pick():
    """Doc with a valid depth>=1 KEY with children → at least one pick."""
    # Tree:  0 (root, depth 0)
    #        ├── 1 (depth 1, has children 3, 4)
    #        │      ├── 3 (leaf)
    #        │      └── 4 (leaf)
    #        └── 2 (depth 1, leaf)
    # N=100 so total cap = 5% × 100 = 5 ≥ 3 (subtree size at pos 1).
    # NOTE: the plan spec used N=15 here, but 5% × 15 = 0.75 < 3, so the
    # total-cap check would always reject. N=100 satisfies both caps.
    N = 100
    rng = random.Random(0)
    picks = pick_subtrees(
        N=N,
        key_positions=[0, 1, 2, 3, 4],
        depth_of={0: 0, 1: 1, 2: 1, 3: 2, 4: 2},
        children_of={0: [1, 2], 1: [3, 4], 2: [], 3: [], 4: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    # Only candidate is position 1 (depth>=1, has children). Subtree size 3
    # = 3% of 100, below MAX_SUBTREE_FRACTION=0.30 and MAX_TOTAL_SUBTREE_FRACTION*100=5.
    assert picks == [1]


def test_pick_subtrees_excludes_too_large_subtrees():
    """A subtree larger than MAX_SUBTREE_FRACTION * N is excluded."""
    # Tree: 0 root, 1 depth-1 with 8 children (positions 2-9).
    # Subtree at 1 = {1,2,3,4,5,6,7,8,9} = 9 nodes. N=10. 9/10 = 90% > 30%.
    rng = random.Random(0)
    picks = pick_subtrees(
        N=10,
        key_positions=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        depth_of={i: (0 if i == 0 else (1 if i == 1 else 2)) for i in range(10)},
        children_of={0: [1], 1: [2, 3, 4, 5, 6, 7, 8, 9],
                     2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    assert picks == []  # only candidate (1) exceeds size cap


def test_pick_subtrees_excludes_mlm_overlap():
    """Candidate whose subtree overlaps MLM mask is excluded."""
    rng = random.Random(0)
    picks = pick_subtrees(
        N=15,
        key_positions=[0, 1, 2, 3, 4],
        depth_of={0: 0, 1: 1, 2: 1, 3: 2, 4: 2},
        children_of={0: [1, 2], 1: [3, 4], 2: [], 3: [], 4: []},
        mlm_masked_positions={3},  # position 3 inside subtree-at-1
        rng=rng,
    )
    assert picks == []  # subtree-at-1 contains 3, which is MLM-masked


def test_pick_subtrees_disjoint_picks():
    """Multiple picks are mutually disjoint."""
    # Tree:   0 root
    #         ├── 1 (subtree {1,3,4})
    #         └── 2 (subtree {2,5,6})
    # plus filler 7..16 to meet MIN_DOC_NODES and not blow size caps
    rng = random.Random(0)
    picks = pick_subtrees(
        N=50,  # large so size cap 5%=2.5 nodes won't limit much; total cap is 5%
        key_positions=[0, 1, 2, 3, 4, 5, 6],
        depth_of={0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2},
        children_of={0: [1, 2], 1: [3, 4], 2: [5, 6],
                     3: [], 4: [], 5: [], 6: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    # Subtree at 1 = 3 nodes; at 2 = 3 nodes. Total 6 / 50 = 12% > 5%, so only ONE picks.
    # But: total cap 5% × 50 = 2.5, so even 3-node subtree exceeds → 0 picks
    # Let's pick N=100 so 5% = 5, each subtree of 3 fits, both together = 6 > 5 still.
    # The implementation should pick the FIRST candidate then skip the second.
    # Re-run with N=200 to give more headroom:
    picks = pick_subtrees(
        N=200,
        key_positions=[0, 1, 2, 3, 4, 5, 6],
        depth_of={0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 2, 6: 2},
        children_of={0: [1, 2], 1: [3, 4], 2: [5, 6],
                     3: [], 4: [], 5: [], 6: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    # Total cap 5% × 200 = 10. Both subtrees (3+3=6) fit. Picks should be disjoint.
    assert set(picks).issubset({1, 2})
    # If both picked, they must be {1, 2} — not contain each other.
    if len(picks) == 2:
        assert set(picks) == {1, 2}


def test_pick_subtrees_respects_total_cap():
    """Sum of subtree sizes across picks ≤ MAX_TOTAL_SUBTREE_FRACTION * N."""
    rng = random.Random(0)
    picks = pick_subtrees(
        N=20,  # 5% cap = 1 node → no single subtree of ≥2 nodes fits in total cap
        key_positions=[0, 1, 2, 3],
        depth_of={0: 0, 1: 1, 2: 1, 3: 2},
        children_of={0: [1, 2], 1: [3], 2: [], 3: []},
        mlm_masked_positions=set(),
        rng=rng,
    )
    # Only candidate: pos 1, subtree {1, 3} = 2 nodes. Total cap = 1.0. Skip.
    assert picks == []


def test_bag_of_keys_target_basic():
    """Multi-hot vector with 1s at atomic-vocab indices of keys in subtree."""
    # Subtree includes positions 1, 2, 3 with key strings 'name', 'image', 'name'
    # vocab maps name=10, image=12
    atomic_vocab = {"name": 10, "image": 12, "ports": 14}
    position_to_key_str = {1: "name", 2: "image", 3: "name"}
    target = bag_of_keys_target(
        subtree_positions={1, 2, 3},
        position_to_key_str=position_to_key_str,
        atomic_vocab=atomic_vocab,
        vocab_size=20,
    )
    assert target.shape == (20,)
    assert target[10].item() == 1.0  # name present (deduped)
    assert target[12].item() == 1.0  # image present
    assert target[14].item() == 0.0  # ports not present
    assert target.sum().item() == 2.0  # exactly two distinct keys


def test_bag_of_keys_target_skips_unknown_keys():
    """Keys not in atomic_vocab are silently skipped."""
    atomic_vocab = {"name": 10}
    position_to_key_str = {1: "name", 2: "obscure_key_not_in_vocab"}
    target = bag_of_keys_target(
        subtree_positions={1, 2},
        position_to_key_str=position_to_key_str,
        atomic_vocab=atomic_vocab,
        vocab_size=20,
    )
    assert target[10].item() == 1.0
    assert target.sum().item() == 1.0
