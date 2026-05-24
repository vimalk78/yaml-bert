"""Test that status-side trigrams are exempt from min_freq filtering.

Background: training corpus is heavily biased against status (only 0.8%
of trigram occurrences are status-side). Normal min_freq filtering drops
nearly all of them. We exempt status-prefixed trigrams so the model has
a chance to learn rare-but-real status fields like
`Deployment::status::replicas`. See scripts/count_status_trigrams.py.
"""
from yaml_bert.linearizer import YamlLinearizer
from yaml_bert.vocab import VocabBuilder, _is_status_trigram


def test_is_status_trigram_helper():
    assert _is_status_trigram("Deployment::status::replicas")
    assert _is_status_trigram("Pod::status::phase")
    assert not _is_status_trigram("Deployment::spec::replicas")
    assert not _is_status_trigram("status")              # too short
    assert not _is_status_trigram("foo")                 # not a trigram


def test_status_trigram_exempt_from_min_freq():
    """A status-side trigram seen once should survive min_freq=100."""
    linearizer = YamlLinearizer()
    # ONE document with a status field — frequency = 1
    nodes = linearizer.linearize("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
status:
  replicas: 3
  readyReplicas: 2
""")
    builder = VocabBuilder()
    vocab = builder.build(nodes, min_freq=100)

    # Status trigrams should survive despite frequency=1
    assert "Deployment::status::replicas" in vocab.kind_target_vocab
    assert "Deployment::status::readyReplicas" in vocab.kind_target_vocab


def test_non_status_trigram_still_filtered():
    """Sanity: non-status trigrams seen once should NOT survive min_freq=100."""
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
""")
    builder = VocabBuilder()
    vocab = builder.build(nodes, min_freq=100)

    # Deployment::spec::replicas was seen once, below threshold → dropped
    assert "Deployment::spec::replicas" not in vocab.kind_target_vocab


def test_per_category_min_freq():
    """Per-category min_freq lets targets (naturally rarer) use a lower
    threshold than input vocabs."""
    linearizer = YamlLinearizer()
    nodes = linearizer.linearize("""\
apiVersion: apps/v1
kind: Deployment
metadata:
  name: web
spec:
  replicas: 3
""")
    builder = VocabBuilder()

    # With tight input filter (100) but lax kind_target filter (1), the
    # spec.replicas trigram should survive — it was seen once.
    vocab = builder.build(
        nodes,
        key_min_freq=100,
        value_min_freq=100,
        simple_target_min_freq=1,   # let bigrams through too
        kind_target_min_freq=1,     # allow seen-once trigrams
    )
    assert "Deployment::spec::replicas" in vocab.kind_target_vocab
