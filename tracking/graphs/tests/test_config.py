from pathlib import Path

from graphs.config import GraphConfig, load_config
from graphs.prescription import ConnectionRules, FullyConnected, Regional

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


def test_defaults_with_no_config():
    config = load_config(None)
    assert config == GraphConfig()
    assert config.prescription == FullyConnected()


def test_load_fully_connected_yaml():
    config = load_config(CONFIGS / "fully_connected.yaml")
    assert config.prescription == FullyConnected(directed=False)


def test_load_regional_yaml():
    config = load_config(CONFIGS / "regional.yaml")
    assert config.prescription == Regional(phi_width=0.5)


def test_load_connection_rules_yaml():
    config = load_config(CONFIGS / "connection_rules.yaml")
    assert config.prescription == ConnectionRules(delta_layer_id=(1.0, 2.0), delta_r=(0.0, 100.0), delta_phi=(-0.3, 0.3))


def test_load_config_with_no_prescription_key_falls_back_to_default(tmp_path):
    empty = tmp_path / "empty.yaml"
    empty.write_text("n_events: 1\n")
    config = load_config(empty)
    assert config.prescription == FullyConnected()
