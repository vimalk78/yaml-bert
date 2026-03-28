from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.types import NodeType


def test_simple_key_value_pairs():
    yaml_str = "app: redis\nrole: replica\ntier: backend\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert len(nodes) == 6

    assert nodes[0].token == "app"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[0].depth == 0
    assert nodes[0].sibling_index == 0
    assert nodes[0].parent_path == ""

    assert nodes[1].token == "redis"
    assert nodes[1].node_type == NodeType.VALUE
    assert nodes[1].depth == 0
    assert nodes[1].sibling_index == 0
    assert nodes[1].parent_path == "app"

    assert nodes[2].token == "role"
    assert nodes[2].node_type == NodeType.KEY
    assert nodes[2].sibling_index == 1

    assert nodes[3].token == "replica"
    assert nodes[3].node_type == NodeType.VALUE
    assert nodes[3].parent_path == "role"

    assert nodes[4].token == "tier"
    assert nodes[4].sibling_index == 2

    assert nodes[5].token == "backend"
    assert nodes[5].parent_path == "tier"
