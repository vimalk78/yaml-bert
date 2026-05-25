from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import Vocabulary, VocabBuilder


def test_extract_parent_key():
    vocab = VocabBuilder().build([])

    assert vocab.extract_parent_key("spec.template.spec.containers.0") == "containers"
    assert vocab.extract_parent_key("metadata") == "metadata"
    assert vocab.extract_parent_key("spec.containers.0.ports.1") == "ports"
    assert vocab.extract_parent_key("") == ""
    assert vocab.extract_parent_key("spec") == "spec"
    assert vocab.extract_parent_key("spec.containers.0") == "containers"
    assert vocab.extract_parent_key("args.0") == "args"


def test_encode_parent_key():
    yaml_str = """\
spec:
  replicas: 3
status:
  replicas: 2
"""
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    builder = VocabBuilder()
    vocab = builder.build(nodes)

    replicas_under_spec = nodes[1]
    parent_key = vocab.extract_parent_key(replicas_under_spec.parent_path)
    assert parent_key == "spec"
    parent_key_id = vocab.encode_key(parent_key)
    assert parent_key_id != vocab.special_tokens["[UNK]"]

    replicas_under_status = nodes[4]
    parent_key = vocab.extract_parent_key(replicas_under_status.parent_path)
    assert parent_key == "status"
    parent_key_id = vocab.encode_key(parent_key)
    assert parent_key_id != vocab.special_tokens["[UNK]"]


def test_encode_parent_key_root_nodes():
    yaml_str = """\
apiVersion: v1
"""
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize(yaml_str)

    builder = VocabBuilder()
    vocab = builder.build(nodes)

    api_key = nodes[0]
    parent_key = vocab.extract_parent_key(api_key.parent_path)
    assert parent_key == ""
    parent_key_id = vocab.encode_key(parent_key)
    assert parent_key_id == vocab.special_tokens["[UNK]"]
