import glob
import os

from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.annotator import DomainAnnotator
from yaml_bert.vocab import VocabBuilder
from yaml_bert.types import NodeType


TEMPLATES_DIR = os.path.join(
    os.path.dirname(__file__), "..", "kubernetes-yaml-templates"
)


def _load_all_nodes():
    """Helper: linearize all YAML files from the real corpus."""
    linearizer = YamlLinearizer()
    yaml_files = glob.glob(
        os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True
    )
    all_nodes = []
    for path in yaml_files:
        all_nodes.extend(linearizer.linearize_file(path))
    return all_nodes


def test_full_pipeline_on_corpus():
    """End-to-end: linearize -> annotate -> build vocab from all YAML files."""
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    builder = VocabBuilder()

    yaml_files = glob.glob(
        os.path.join(TEMPLATES_DIR, "**", "*.yaml"), recursive=True
    )

    all_nodes = []
    for path in yaml_files:
        nodes = linearizer.linearize_file(path)
        annotated = annotator.annotate(nodes)
        all_nodes.extend(annotated)

    vocab = builder.build(all_nodes)

    for key in ["apiVersion", "kind", "metadata", "name", "spec"]:
        assert key in vocab.key_vocab, f"Expected '{key}' in key_vocab"

    for value in ["v1", "Pod"]:
        assert value in vocab.value_vocab, f"Expected '{value}' in value_vocab"

    assert len(vocab.key_vocab) > 20, "Expected 20+ unique keys across corpus"
    assert len(vocab.value_vocab) > 20, "Expected 20+ unique values across corpus"

    for key in ["apiVersion", "kind", "metadata"]:
        assert vocab.decode_key(vocab.encode_key(key)) == key


def test_full_pipeline_single_deployment():
    """Pipeline test using a specific real deployment file."""
    linearizer = YamlLinearizer()
    annotator = DomainAnnotator()
    builder = VocabBuilder()

    path = os.path.join(TEMPLATES_DIR, "deployment", "deployment-nginx.yaml")
    nodes = linearizer.linearize_file(path)
    annotated = annotator.annotate(nodes)
    vocab = builder.build(annotated)

    containers_node = next(n for n in annotated if n.token == "containers")
    assert containers_node.annotations["list_ordered"] is False

    assert "apiVersion" in vocab.key_vocab
    assert "Deployment" in vocab.value_vocab
    assert "containers" in vocab.key_vocab

    assert vocab.encode_key("apiVersion") >= 3
    assert vocab.encode_value("Deployment") >= 3


def test_spec_at_two_depths():
    """'spec' at two different depths gets the same token ID but different parent_paths."""
    linearizer = YamlLinearizer()
    path = os.path.join(TEMPLATES_DIR, "deployment", "deployment-nginx.yaml")
    nodes = linearizer.linearize_file(path)

    spec_nodes = [n for n in nodes if n.token == "spec"]
    assert len(spec_nodes) == 2

    assert spec_nodes[0].token == spec_nodes[1].token
    assert spec_nodes[0].depth != spec_nodes[1].depth
    assert spec_nodes[0].parent_path != spec_nodes[1].parent_path


def test_vocab_save_load_roundtrip_on_corpus(tmp_path):
    """Build vocab from real corpus, save, reload, verify identical."""
    all_nodes = _load_all_nodes()

    builder = VocabBuilder()
    vocab = builder.build(all_nodes)

    vocab_path = str(tmp_path / "corpus_vocab.json")
    vocab.save(vocab_path)

    from yaml_bert.vocab import Vocabulary
    loaded = Vocabulary.load(vocab_path)

    assert loaded.key_vocab == vocab.key_vocab
    assert loaded.value_vocab == vocab.value_vocab
    assert loaded.special_tokens == vocab.special_tokens
