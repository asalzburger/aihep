from pathlib import Path

from detectorreco2d.config import RecoConfig, Resolution, load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIG_PATH = CONFIGS / "default.yaml"


def test_defaults_with_no_config_means_no_smearing():
    config = load_config(None)
    assert config == RecoConfig()
    assert config.track.d0 == Resolution()
    assert config.track.phi0 == Resolution()
    assert config.track.pt == Resolution()
    assert config.cluster.energy == Resolution()
    assert config.seed is None


def test_load_default_yaml():
    config = load_config(DEFAULT_CONFIG_PATH)
    assert config.track.d0 == Resolution(a=0.05, b=2.0)
    assert config.track.phi0 == Resolution(a=0.001, b=0.02)
    assert config.track.pt == Resolution(a=0.05, b=1.0)
    assert config.cluster.energy == Resolution(a=0.05, b=2.0)
    assert config.seed == 7


def test_load_config_missing_quantities_default_to_no_smearing(tmp_path):
    path = tmp_path / "partial.yaml"
    path.write_text("track_resolution:\n  d0: {a: 1.0, b: 2.0}\n")
    config = load_config(path)
    assert config.track.d0 == Resolution(a=1.0, b=2.0)
    assert config.track.phi0 == Resolution()
    assert config.track.pt == Resolution()
    assert config.cluster.energy == Resolution()
    assert config.seed is None
