from yaml_bert.config import YamlBertConfig


def test_default_config():
    config = YamlBertConfig()

    assert config.d_model == 256
    assert config.num_layers == 6
    assert config.num_heads == 8
    assert config.d_ff == 1024
    assert config.max_depth == 16
    assert config.max_sibling == 32
    assert config.mask_prob == 0.15
    assert config.lr == 1e-4
    assert config.batch_size == 32
    assert config.num_epochs == 30
    assert config.max_seq_len == 512


def test_custom_config():
    config = YamlBertConfig(d_model=64, num_layers=2, num_heads=2)

    assert config.d_model == 64
    assert config.num_layers == 2
    assert config.num_heads == 2
    assert config.d_ff == 256


def test_d_ff_defaults_to_4x_d_model():
    config = YamlBertConfig(d_model=128)
    assert config.d_ff == 512

    config_custom = YamlBertConfig(d_model=128, d_ff=1024)
    assert config_custom.d_ff == 1024


def test_auxiliary_loss_weights():
    config = YamlBertConfig()
    assert config.aux_kind_weight == 0.1
    assert config.aux_parent_weight == 0.1

    config2 = YamlBertConfig(aux_kind_weight=0.5, aux_parent_weight=0.0)
    assert config2.aux_kind_weight == 0.5
    assert config2.aux_parent_weight == 0.0
