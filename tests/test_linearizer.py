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


def test_nested_mapping():
    yaml_str = "metadata:\n  name: nginx\n  namespace: default\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert len(nodes) == 5

    assert nodes[0].token == "metadata"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[0].depth == 0
    assert nodes[0].sibling_index == 0
    assert nodes[0].parent_path == ""

    assert nodes[1].token == "name"
    assert nodes[1].node_type == NodeType.KEY
    assert nodes[1].depth == 1
    assert nodes[1].sibling_index == 0
    assert nodes[1].parent_path == "metadata"

    assert nodes[2].token == "nginx"
    assert nodes[2].node_type == NodeType.VALUE
    assert nodes[2].depth == 1
    assert nodes[2].sibling_index == 0
    assert nodes[2].parent_path == "metadata.name"

    assert nodes[3].token == "namespace"
    assert nodes[3].node_type == NodeType.KEY
    assert nodes[3].depth == 1
    assert nodes[3].sibling_index == 1
    assert nodes[3].parent_path == "metadata"

    assert nodes[4].token == "default"
    assert nodes[4].node_type == NodeType.VALUE
    assert nodes[4].depth == 1
    assert nodes[4].sibling_index == 1
    assert nodes[4].parent_path == "metadata.namespace"


def test_list_of_maps():
    yaml_str = (
        "containers:\n"
        "- name: webserver1\n"
        "  image: nginx:1.6\n"
        "- name: database-server\n"
        "  image: mysql-3.2\n"
    )
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert len(nodes) == 9

    assert nodes[0].token == "containers"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[0].depth == 0
    assert nodes[0].parent_path == ""

    assert nodes[1].token == "name"
    assert nodes[1].node_type == NodeType.LIST_KEY
    assert nodes[1].depth == 1
    assert nodes[1].sibling_index == 0
    assert nodes[1].parent_path == "containers.0"

    assert nodes[2].token == "webserver1"
    assert nodes[2].node_type == NodeType.LIST_VALUE
    assert nodes[2].depth == 1
    assert nodes[2].sibling_index == 0
    assert nodes[2].parent_path == "containers.0.name"

    assert nodes[3].token == "image"
    assert nodes[3].node_type == NodeType.LIST_KEY
    assert nodes[3].depth == 1
    assert nodes[3].sibling_index == 1
    assert nodes[3].parent_path == "containers.0"

    assert nodes[4].token == "nginx:1.6"
    assert nodes[4].node_type == NodeType.LIST_VALUE
    assert nodes[4].depth == 1
    assert nodes[4].sibling_index == 1
    assert nodes[4].parent_path == "containers.0.image"

    assert nodes[5].token == "name"
    assert nodes[5].node_type == NodeType.LIST_KEY
    assert nodes[5].depth == 1
    assert nodes[5].sibling_index == 0
    assert nodes[5].parent_path == "containers.1"

    assert nodes[6].token == "database-server"
    assert nodes[6].node_type == NodeType.LIST_VALUE
    assert nodes[6].depth == 1
    assert nodes[6].sibling_index == 0
    assert nodes[6].parent_path == "containers.1.name"

    assert nodes[7].token == "image"
    assert nodes[7].node_type == NodeType.LIST_KEY
    assert nodes[7].depth == 1
    assert nodes[7].sibling_index == 1
    assert nodes[7].parent_path == "containers.1"

    assert nodes[8].token == "mysql-3.2"
    assert nodes[8].node_type == NodeType.LIST_VALUE
    assert nodes[8].depth == 1
    assert nodes[8].sibling_index == 1
    assert nodes[8].parent_path == "containers.1.image"


def test_scalar_list():
    yaml_str = "args:\n- --config\n- /etc/app.yaml\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert len(nodes) == 3

    assert nodes[0].token == "args"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[0].depth == 0

    assert nodes[1].token == "--config"
    assert nodes[1].node_type == NodeType.LIST_VALUE
    assert nodes[1].depth == 1
    assert nodes[1].sibling_index == 0
    assert nodes[1].parent_path == "args.0"

    assert nodes[2].token == "/etc/app.yaml"
    assert nodes[2].node_type == NodeType.LIST_VALUE
    assert nodes[2].depth == 1
    assert nodes[2].sibling_index == 1
    assert nodes[2].parent_path == "args.1"


def test_nested_list_in_list_item():
    yaml_str = (
        "containers:\n"
        "- name: webserver1\n"
        "  ports:\n"
        "  - containerPort: 80\n"
        "  - containerPort: 443\n"
    )
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert len(nodes) == 8

    assert nodes[0].token == "containers"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[0].depth == 0

    assert nodes[1].token == "name"
    assert nodes[1].node_type == NodeType.LIST_KEY
    assert nodes[1].depth == 1
    assert nodes[1].parent_path == "containers.0"

    assert nodes[2].token == "webserver1"
    assert nodes[2].node_type == NodeType.LIST_VALUE
    assert nodes[2].parent_path == "containers.0.name"

    assert nodes[3].token == "ports"
    assert nodes[3].node_type == NodeType.LIST_KEY
    assert nodes[3].depth == 1
    assert nodes[3].parent_path == "containers.0"

    assert nodes[4].token == "containerPort"
    assert nodes[4].node_type == NodeType.LIST_KEY
    assert nodes[4].depth == 2
    assert nodes[4].parent_path == "containers.0.ports.0"

    assert nodes[5].token == "80"
    assert nodes[5].node_type == NodeType.LIST_VALUE
    assert nodes[5].depth == 2
    assert nodes[5].parent_path == "containers.0.ports.0.containerPort"

    assert nodes[6].token == "containerPort"
    assert nodes[6].node_type == NodeType.LIST_KEY
    assert nodes[6].depth == 2
    assert nodes[6].parent_path == "containers.0.ports.1"

    assert nodes[7].token == "443"
    assert nodes[7].node_type == NodeType.LIST_VALUE
    assert nodes[7].depth == 2
    assert nodes[7].parent_path == "containers.0.ports.1.containerPort"
