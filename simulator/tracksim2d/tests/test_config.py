from pathlib import Path

from detector2d.geometry import CircleLayer, LineLayer
from tracksim2d.config import SimConfig, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"


def test_defaults_with_no_config():
    config = load_config(None)
    assert config == SimConfig()
    assert config.layers == []
    assert config.n_events == 1


def test_load_default_yaml_parses_layers_and_gun():
    config = load_config(CONFIG_PATH)
    assert len(config.layers) == 6
    assert isinstance(config.layers[0], LineLayer)
    assert config.layers[0].p1 == (10.0, -50.0)
    assert config.layers[0].pitch == 1.0
    assert isinstance(config.layers[-1], CircleLayer)
    assert config.layers[-1].radius == 5.0

    assert config.magnetic_field.bz == 1.0
    assert config.gun.n_particles == 3
    assert config.gun.charges == (-1.0, 1.0)
    assert config.n_events == 1
