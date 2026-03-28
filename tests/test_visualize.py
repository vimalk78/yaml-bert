import os

from yaml_bert.visualize import plot_training_loss, plot_embedding_similarity, plot_attention_patterns


def test_plot_training_loss(tmp_path):
    losses = [5.2, 4.8, 4.1, 3.5, 3.0, 2.7, 2.4, 2.2, 2.0, 1.9]
    output_path = str(tmp_path / "loss.png")

    plot_training_loss(losses, output_path=output_path)

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_plot_embedding_similarity(tmp_path):
    embedding_results = [
        {
            "key": "spec",
            "position_a": {"depth": 0, "parent_key": ""},
            "position_b": {"depth": 2, "parent_key": "template"},
            "cosine_similarity": 0.45,
        },
        {
            "key": "name",
            "position_a": {"depth": 1, "parent_key": "metadata"},
            "position_b": {"depth": 1, "parent_key": "containers"},
            "cosine_similarity": 0.32,
        },
    ]
    output_path = str(tmp_path / "embeddings.png")

    plot_embedding_similarity(embedding_results, output_path=output_path)

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0


def test_plot_attention_patterns(tmp_path):
    import torch
    attention_weights = torch.rand(2, 8, 8)
    token_labels = ["apiVersion", "apps/v1", "kind", "Deployment",
                    "metadata", "name", "nginx", "spec"]
    output_path = str(tmp_path / "attention.png")

    plot_attention_patterns(
        attention_weights,
        token_labels=token_labels,
        output_path=output_path,
    )

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0
