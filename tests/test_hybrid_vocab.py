from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder, Vocabulary, compute_target
from yaml_bert.types import YamlNode, NodeType


def test_compute_target_unigram():
    node = YamlNode("metadata", NodeType.KEY, depth=0, sibling_index=2, parent_path="")
    assert compute_target(node, "Deployment") == ("metadata", "simple")

    node = YamlNode("spec", NodeType.KEY, depth=0, sibling_index=3, parent_path="")
    assert compute_target(node, "Deployment") == ("spec", "simple")


def test_compute_target_bigram_metadata():
    node = YamlNode("name", NodeType.KEY, depth=1, sibling_index=0, parent_path="metadata")
    assert compute_target(node, "Deployment") == ("metadata::name", "simple")

    node = YamlNode("labels", NodeType.KEY, depth=1, sibling_index=1, parent_path="metadata")
    assert compute_target(node, "Deployment") == ("metadata::labels", "simple")


def test_compute_target_trigram_spec():
    node = YamlNode("replicas", NodeType.KEY, depth=1, sibling_index=0, parent_path="spec")
    assert compute_target(node, "Deployment") == ("Deployment::spec::replicas", "kind_specific")

    node = YamlNode("ports", NodeType.LIST_KEY, depth=1, sibling_index=0, parent_path="spec")
    assert compute_target(node, "Service") == ("Service::spec::ports", "kind_specific")


def test_compute_target_trigram_data():
    node = YamlNode("DB_HOST", NodeType.KEY, depth=1, sibling_index=0, parent_path="data")
    assert compute_target(node, "ConfigMap") == ("ConfigMap::data::DB_HOST", "kind_specific")


def test_compute_target_trigram_rules():
    node = YamlNode("apiGroups", NodeType.LIST_KEY, depth=1, sibling_index=0, parent_path="rules")
    assert compute_target(node, "ClusterRole") == ("ClusterRole::rules::apiGroups", "kind_specific")


def test_compute_target_bigram_deeper():
    node = YamlNode("image", NodeType.LIST_KEY, depth=4, sibling_index=1, parent_path="spec.template.spec.containers.0")
    assert compute_target(node, "Deployment") == ("containers::image", "simple")

    node = YamlNode("containerPort", NodeType.LIST_KEY, depth=5, sibling_index=0, parent_path="spec.template.spec.containers.0.ports.0")
    assert compute_target(node, "Deployment") == ("ports::containerPort", "simple")


def test_build_hybrid_vocabs():
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes = linearizer.linearize("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
  labels:
    app: web
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
""")
    annotator.annotate(nodes)
    vocab = VocabBuilder().build(nodes)

    assert "metadata::name" in vocab.simple_target_vocab
    assert "metadata::labels" in vocab.simple_target_vocab
    assert "Deployment::spec::replicas" in vocab.kind_target_vocab
    assert "Deployment::spec::selector" in vocab.kind_target_vocab


def test_encode_targets():
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
  - name: app
""")
    annotator.annotate(nodes)
    vocab = VocabBuilder().build(nodes)

    assert vocab.encode_simple_target("metadata::name") != vocab.special_tokens["[UNK]"]
    assert vocab.encode_kind_target("Pod::spec::containers") != vocab.special_tokens["[UNK]"]
    assert vocab.encode_simple_target("nonexistent") == vocab.special_tokens["[UNK]"]


def test_hybrid_vocab_saved_loaded(tmp_path):
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Pod
metadata:
  name: test
spec:
  containers:
  - name: app
""")
    annotator.annotate(nodes)
    vocab = VocabBuilder().build(nodes)

    path = str(tmp_path / "vocab.json")
    vocab.save(path)
    loaded = Vocabulary.load(path)

    assert loaded.simple_target_vocab == vocab.simple_target_vocab
    assert loaded.kind_target_vocab == vocab.kind_target_vocab


def test_vocab_sizes():
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    nodes = linearizer.linearize("""\
apiVersion: v1
kind: Service
metadata:
  name: svc
spec:
  ports:
  - port: 80
  type: ClusterIP
""")
    annotator.annotate(nodes)
    vocab = VocabBuilder().build(nodes)

    assert vocab.simple_target_vocab_size > 0
    assert vocab.kind_target_vocab_size > 0
