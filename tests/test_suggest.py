import torch
from yaml_bert.config import YamlBertConfig
from yaml_bert.embedding import YamlBertEmbedding
from yaml_bert.model import YamlBertModel
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder
from yaml_bert.suggest import suggest_missing_fields


def _build_model_and_vocab():
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()

    yaml_with_all_fields = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
  namespace: default
  labels:
    app: web
  annotations:
    description: test
spec:
  replicas: 3
  selector:
    matchLabels:
      app: web
  template:
    metadata:
      labels:
        app: web
    spec:
      containers:
      - name: web
        image: nginx
        ports:
        - containerPort: 80
        resources:
          limits:
            memory: 128Mi
          requests:
            memory: 64Mi
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
"""
    nodes = linearizer.linearize(yaml_with_all_fields)
    annotator.annotate(nodes)
    vocab = VocabBuilder().build(nodes)

    config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)
    emb = YamlBertEmbedding(
        config=config,
        key_vocab_size=vocab.key_vocab_size,
        value_vocab_size=vocab.value_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    model = YamlBertModel(
        config=config, embedding=emb,
        key_vocab_size=vocab.key_vocab_size,
        kind_vocab_size=vocab.kind_vocab_size,
    )
    return model, vocab


def test_suggest_returns_list():
    model, vocab = _build_model_and_vocab()
    yaml_text = """\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: app
spec:
  replicas: 3
"""
    suggestions = suggest_missing_fields(model, vocab, yaml_text)
    assert isinstance(suggestions, list)


def test_suggest_each_item_has_required_keys():
    model, vocab = _build_model_and_vocab()
    yaml_text = """\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
"""
    suggestions = suggest_missing_fields(model, vocab, yaml_text, threshold=0.1)
    for s in suggestions:
        assert "parent_path" in s
        assert "missing_key" in s
        assert "confidence" in s
        assert 0.0 <= s["confidence"] <= 1.0


def test_suggest_sorted_by_confidence():
    model, vocab = _build_model_and_vocab()
    yaml_text = """\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
"""
    suggestions = suggest_missing_fields(model, vocab, yaml_text, threshold=0.1)
    if len(suggestions) > 1:
        for i in range(len(suggestions) - 1):
            assert suggestions[i]["confidence"] >= suggestions[i + 1]["confidence"]


def test_suggest_does_not_report_existing_keys():
    model, vocab = _build_model_and_vocab()
    yaml_text = """\
apiVersion: v1
kind: Pod
metadata:
  name: pod
spec:
  containers:
  - name: app
    image: nginx
"""
    suggestions = suggest_missing_fields(model, vocab, yaml_text, threshold=0.01)
    from yaml_bert.linearizer import YamlLinearizer
    from yaml_bert.annotator import DomainAnnotator
    from yaml_bert.types import NodeType
    nodes = YamlLinearizer().linearize(yaml_text)
    DomainAnnotator().annotate(nodes)
    existing_keys_by_parent: dict[str, set[str]] = {}
    for n in nodes:
        if n.node_type in (NodeType.KEY, NodeType.LIST_KEY):
            existing_keys_by_parent.setdefault(n.parent_path, set()).add(n.token)

    for s in suggestions:
        parent = s["parent_path"]
        if parent in existing_keys_by_parent:
            assert s["missing_key"] not in existing_keys_by_parent[parent], \
                f"Reported '{s['missing_key']}' as missing but it exists at {parent}"
