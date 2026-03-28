from yaml_bert.annotator import DomainAnnotator
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.types import NodeType


def test_annotate_unordered_list():
    yaml_str = (
        "spec:\n"
        "  containers:\n"
        "  - name: nginx\n"
        "  - name: sidecar\n"
    )
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    nodes = linearizer.linearize(yaml_str)
    annotated = annotator.annotate(nodes)

    containers_node = next(n for n in annotated if n.token == "containers")
    assert containers_node.annotations["list_ordered"] is False


def test_annotate_ordered_list():
    yaml_str = (
        "spec:\n"
        "  initContainers:\n"
        "  - name: init-db\n"
        "  - name: init-cache\n"
    )
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    nodes = linearizer.linearize(yaml_str)
    annotated = annotator.annotate(nodes)

    init_node = next(n for n in annotated if n.token == "initContainers")
    assert init_node.annotations["list_ordered"] is True


def test_non_list_keys_have_no_annotation():
    yaml_str = "apiVersion: v1\nkind: Pod\n"
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    nodes = linearizer.linearize(yaml_str)
    annotated = annotator.annotate(nodes)

    for node in annotated:
        assert "list_ordered" not in node.annotations
