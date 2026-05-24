import torch

from yaml_bert.config import YamlBertConfig
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.v8_dataset import V8Dataset, compute_children_info, v8_collate_fn


def test_compute_children_info_simple():
    """For 'spec: {replicas: 3}', spec has 1 child (replicas)."""
    nodes = YamlLinearizer().linearize("spec:\n  replicas: 3\n")
    info = compute_children_info(nodes)

    # info["children_of"][i] = list of positions that are children of position i
    # info["parent_of"][i] = position of i's parent (or -1 if root)
    # info["key_positions"] = sorted list of positions whose node_type is KEY/LIST_KEY
    # info["depth_of"][i] = depth at position i

    assert "spec" in [nodes[p].token for p in info["key_positions"]]
    assert "replicas" in [nodes[p].token for p in info["key_positions"]]

    # spec is at position 0, has children: replicas (pos 1)
    spec_pos = next(p for p in info["key_positions"] if nodes[p].token == "spec")
    assert nodes[info["children_of"][spec_pos][0]].token == "replicas"


def test_compute_children_info_nested():
    """For nested structure, children_of correctly tracks parent-child."""
    yaml_str = """\
spec:
  selector:
    matchLabels:
      app: nginx
"""
    nodes = YamlLinearizer().linearize(yaml_str)
    info = compute_children_info(nodes)

    by_token = {nodes[p].token: p for p in info["key_positions"]}

    spec_pos = by_token["spec"]
    selector_pos = by_token["selector"]
    matchLabels_pos = by_token["matchLabels"]
    app_pos = by_token["app"]

    assert info["children_of"][spec_pos] == [selector_pos]
    assert info["children_of"][selector_pos] == [matchLabels_pos]
    assert info["children_of"][matchLabels_pos] == [app_pos]

    assert info["parent_of"][selector_pos] == spec_pos
    assert info["parent_of"][matchLabels_pos] == selector_pos


def test_compute_children_info_list_of_mappings():
    """KEYs inside list items should link to the list-key (not be orphaned).

    For 'spec.containers' as a list of dicts, the linearizer assigns each
    inner KEY a parent_path like 'spec.containers.0'. Since no node has that
    full_path, we strip the trailing numeric segment and link to
    'spec.containers'. All inner KEYs across all items become children of
    the list key.
    """
    yaml_str = """\
spec:
  containers:
  - name: a
    image: x
  - name: b
    image: y
"""
    nodes = YamlLinearizer().linearize(yaml_str)
    info = compute_children_info(nodes)

    # Find the list-key 'containers' (there should be exactly one)
    containers_positions = [
        p for p in info["key_positions"] if nodes[p].token == "containers"
    ]
    assert len(containers_positions) == 1
    containers_pos = containers_positions[0]

    # Find each name/image position. There are two of each (item0 and item1).
    name_positions = [
        p for p in info["key_positions"] if nodes[p].token == "name"
    ]
    image_positions = [
        p for p in info["key_positions"] if nodes[p].token == "image"
    ]
    assert len(name_positions) == 2
    assert len(image_positions) == 2

    # All 4 inner KEYs are children of containers (per-item grouping lost)
    children = info["children_of"][containers_pos]
    for p in name_positions + image_positions:
        assert p in children, (
            f"Position {p} ({nodes[p].token}, parent_path="
            f"{nodes[p].parent_path!r}) is not a child of containers"
        )
        assert info["parent_of"][p] == containers_pos


def test_compute_children_info_list_of_scalars():
    """List of scalars (e.g., args) has no inner KEYs; should not crash.

    The list values are LIST_VALUE leaves, not KEYs, so 'args' has no KEY
    children. This must not crash and must produce a valid (empty) child list.
    """
    yaml_str = """\
args:
  - --foo
  - --bar
"""
    nodes = YamlLinearizer().linearize(yaml_str)
    info = compute_children_info(nodes)

    args_positions = [
        p for p in info["key_positions"] if nodes[p].token == "args"
    ]
    assert len(args_positions) == 1
    args_pos = args_positions[0]

    # args is a root KEY with no KEY children
    assert info["parent_of"][args_pos] == -1
    assert info["children_of"][args_pos] == []


def test_compute_children_info_multi_root():
    """Top-level KEYs at root depth have parent_of == -1."""
    yaml_str = "apiVersion: v1\nkind: Pod\n"
    nodes = YamlLinearizer().linearize(yaml_str)
    info = compute_children_info(nodes)

    by_token = {nodes[p].token: p for p in info["key_positions"]}
    assert "apiVersion" in by_token
    assert "kind" in by_token

    assert info["parent_of"][by_token["apiVersion"]] == -1
    assert info["parent_of"][by_token["kind"]] == -1


def test_compute_children_info_empty_input():
    """Empty node list returns all-empty lists without crashing."""
    info = compute_children_info([])
    assert info["children_of"] == []
    assert info["parent_of"] == []
    assert info["key_positions"] == []
    assert info["depth_of"] == []
    assert info["full_path_of"] == []


def test_v8_dataset_item_keys():
    """V8Dataset item contains the required keys."""
    nodes_list = [
        YamlLinearizer().linearize("apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n")
        for _ in range(2)
    ]
    vocab = __import__("yaml_bert.vocab", fromlist=["VocabBuilder"]).VocabBuilder().build(
        [n for doc in nodes_list for n in doc], min_freq=1,
    )
    config = YamlBertConfig(v8_mode=True, mask_prob=1.0)  # mask all maskable for deterministic test
    ds = V8Dataset(documents=nodes_list, vocab=vocab, config=config)

    item = ds[0]
    assert "token_ids" in item
    assert "node_types" in item
    assert "depths" in item
    assert "sibling_indices" in item
    assert "atomic_labels" in item
    assert "children_info" in item  # dict, not tensor — collate handles


def test_v8_collate_preserves_children_info():
    """Collate returns batch with batched tensors and a list of children_info."""
    nodes_list = [
        YamlLinearizer().linearize("apiVersion: v1\nkind: Pod\n"),
        YamlLinearizer().linearize("apiVersion: v1\nkind: Service\nspec:\n  x: 1\n"),
    ]
    vocab = __import__("yaml_bert.vocab", fromlist=["VocabBuilder"]).VocabBuilder().build(
        [n for doc in nodes_list for n in doc], min_freq=1,
    )
    config = YamlBertConfig(v8_mode=True, mask_prob=0.5)
    ds = V8Dataset(documents=nodes_list, vocab=vocab, config=config)
    batch = v8_collate_fn([ds[0], ds[1]])
    assert batch["token_ids"].dim() == 2  # (B, N)
    assert isinstance(batch["batch_info"], list)
    assert len(batch["batch_info"]) == 2
    assert "children_of" in batch["batch_info"][0]


def test_v8_collate_includes_aggregator_precompute():
    """v8_collate_fn precomputes tensors needed by the vectorized aggregator."""
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.config import YamlBertConfig
    from yaml_bert.vocab import VocabBuilder
    from yaml_bert.v8_dataset import V8Dataset, v8_collate_fn

    docs = [
        YamlLinearizer().linearize("apiVersion: v1\nkind: Pod\nspec:\n  x: 1\n"),
        YamlLinearizer().linearize("apiVersion: v1\nkind: Service\n"),
    ]
    vocab = VocabBuilder().build([n for d in docs for n in d], min_freq=1)
    config = YamlBertConfig(v8_mode=True, mask_prob=0.0)
    ds = V8Dataset(documents=docs, vocab=vocab, config=config)
    batch = v8_collate_fn([ds[0], ds[1]])

    # parent_of_tensor: (B, N) long, -1 sentinel for no-parent or padding
    assert "parent_of_tensor" in batch
    pt = batch["parent_of_tensor"]
    assert pt.dim() == 2
    assert pt.dtype == torch.long
    assert pt.shape[0] == 2  # B
    # Doc 0: "spec" at pos 0 is root → parent_of = -1
    #        "x" at pos 1 is child of spec → parent_of points to spec's index
    # Doc 1: "apiVersion" root → -1, "kind" root → -1

    # top_level_key_mask: (B, N) bool, True at depth-0 KEY positions
    assert "top_level_key_mask" in batch
    tlkm = batch["top_level_key_mask"]
    assert tlkm.dim() == 2
    assert tlkm.dtype == torch.bool
    assert tlkm.shape == pt.shape

    # edges_by_depth: dict[int, tensor (E, 3)] of [doc_idx, child_pos, parent_pos]
    # parents_by_depth: dict[int, tensor (P, 2)] of [doc_idx, parent_pos] with at-least-one-child
    assert "edges_by_depth" in batch
    assert "parents_by_depth" in batch
    assert isinstance(batch["edges_by_depth"], dict)
    assert isinstance(batch["parents_by_depth"], dict)
    # Same set of depth keys in both
    assert set(batch["edges_by_depth"].keys()) == set(batch["parents_by_depth"].keys())

    # Per-depth shape check: edges has (E, 3), parents has (P, 2)
    for d, edges in batch["edges_by_depth"].items():
        assert edges.dim() == 2 and edges.shape[1] == 3
        assert edges.dtype == torch.long
    for d, parents in batch["parents_by_depth"].items():
        assert parents.dim() == 2 and parents.shape[1] == 2
        assert parents.dtype == torch.long
