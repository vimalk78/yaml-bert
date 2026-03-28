import glob
import os

from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.types import NodeType

TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "kubernetes-yaml-templates"
)


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


def test_multi_document():
    yaml_str = "---\nkind: Deployment\n---\nkind: Service\n"
    linearizer = YamlLinearizer()
    docs = linearizer.linearize_multi_doc(yaml_str)

    assert len(docs) == 2

    assert len(docs[0]) == 2
    assert docs[0][0].token == "kind"
    assert docs[0][1].token == "Deployment"

    assert len(docs[1]) == 2
    assert docs[1][0].token == "kind"
    assert docs[1][1].token == "Service"


def test_linearize_file():
    linearizer = YamlLinearizer()
    path = os.path.join(TEMPLATES_DIR, "deployment", "deployment-nginx.yaml")
    nodes = linearizer.linearize_file(path)

    assert len(nodes) > 0

    assert nodes[0].token == "apiVersion"
    assert nodes[0].node_type == NodeType.KEY
    assert nodes[1].token == "apps/v1"
    assert nodes[1].node_type == NodeType.VALUE

    kind_node = next(n for n in nodes if n.token == "kind")
    assert kind_node.node_type == NodeType.KEY
    deployment_node = next(n for n in nodes if n.token == "Deployment")
    assert deployment_node.node_type == NodeType.VALUE


def test_linearize_file_service():
    linearizer = YamlLinearizer()
    path = os.path.join(TEMPLATES_DIR, "service", "service-clusterip-nginx.yaml")
    nodes = linearizer.linearize_file(path)

    assert len(nodes) > 0
    kind_node = next(n for n in nodes if n.token == "kind")
    service_node = next(n for n in nodes if n.token == "Service")
    assert kind_node.node_type == NodeType.KEY
    assert service_node.node_type == NodeType.VALUE


def test_empty_yaml():
    linearizer = YamlLinearizer()
    assert linearizer.linearize("") == []
    assert linearizer.linearize("---") == []


def test_boolean_and_null_values():
    yaml_str = "enabled: true\ncount: 0\nmissing: null\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert nodes[1].token == "True"
    assert nodes[3].token == "0"
    assert nodes[5].token == "None"


def test_deeply_nested():
    yaml_str = "a:\n  b:\n    c:\n      d: leaf\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert nodes[0].token == "a"
    assert nodes[0].depth == 0
    assert nodes[0].parent_path == ""

    assert nodes[1].token == "b"
    assert nodes[1].depth == 1
    assert nodes[1].parent_path == "a"

    assert nodes[2].token == "c"
    assert nodes[2].depth == 2
    assert nodes[2].parent_path == "a.b"

    assert nodes[3].token == "d"
    assert nodes[3].depth == 3
    assert nodes[3].parent_path == "a.b.c"

    assert nodes[4].token == "leaf"
    assert nodes[4].depth == 3
    assert nodes[4].parent_path == "a.b.c.d"


def test_integer_and_float_values():
    yaml_str = "replicas: 3\ncpu: 0.5\n"
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    assert nodes[1].token == "3"
    assert nodes[1].node_type == NodeType.VALUE

    assert nodes[3].token == "0.5"
    assert nodes[3].node_type == NodeType.VALUE


def test_linearize_all_kubernetes_templates():
    """Smoke test: linearize every YAML file in kubernetes-yaml-templates/.
    Ensures no crashes on real-world K8s manifests."""
    linearizer = YamlLinearizer()
    yaml_files = glob.glob(
        os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True
    )
    assert len(yaml_files) > 40, f"Expected 40+ YAML files, found {len(yaml_files)}"

    total_nodes = 0
    for path in yaml_files:
        nodes = linearizer.linearize_file(path)
        assert len(nodes) > 0, f"Empty linearization for {path}"
        for node in nodes:
            assert node.token is not None
            assert node.node_type is not None
            assert node.depth >= 0
            assert node.sibling_index >= 0
        total_nodes += len(nodes)

    assert total_nodes > 500, f"Expected 500+ total nodes, got {total_nodes}"
