from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.v8_dataset import compute_children_info


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
