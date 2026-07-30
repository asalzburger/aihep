from pathlib import Path

import pytest
from detector2d.geometry import CircleLayer, LineLayer
from tracksim2d.config import SimConfig, load_config

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
BARREL6_CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "barrel6.yaml"


def test_defaults_with_no_config():
    config = load_config(None)
    assert config == SimConfig()
    assert config.layers == []
    assert config.n_events == 1
    assert config.tracker_boundary is None


def test_tracker_boundary_loaded_from_yaml(tmp_path):
    config_path = tmp_path / "with_boundary.yaml"
    config_path.write_text("tracker_boundary: 210.0\n")
    config = load_config(config_path)
    assert config.tracker_boundary == 210.0


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


def test_load_barrel6_yaml_detailed_builds_tilted_module_rings():
    config = load_config(BARREL6_CONFIG_PATH)
    # every module is a LineLayer; layer_ids 0-5 each appear many times (one
    # per module in that layer's ring), not once each -- this exercises
    # tracksim2d's delegation to detector2d.config.build_layers_from_raw for
    # the `detector:` form (see detector2d's own tests for the parsing unit
    # tests).
    assert all(isinstance(layer, LineLayer) for layer in config.layers)
    layer_ids = {layer.layer_id for layer in config.layers}
    assert layer_ids == {0, 1, 2, 3, 4, 5}
    per_layer_counts = {lid: sum(1 for layer in config.layers if layer.layer_id == lid) for lid in layer_ids}
    assert all(count > 1 for count in per_layer_counts.values())
    # precision layers (0-2) use the precision module size/pitch, outer (3-5) the outer one
    assert all(layer.pitch == 0.1 for layer in config.layers if layer.layer_id in (0, 1, 2))
    assert all(layer.pitch == 0.5 for layer in config.layers if layer.layer_id in (3, 4, 5))


def test_detector_and_flat_layers_are_mutually_exclusive(tmp_path):
    bad_config = tmp_path / "bad.yaml"
    bad_config.write_text(
        """
        detector:
          mode: simplified
          layers: [{layer_id: 0, radius: 10.0, kind: precision}]
          module_types: {precision: {half_length: 1.0, tilt_deg: 0.0, overlap_fraction: 0.0}}
        layers:
          - {kind: circle, layer_id: 0, center: [0.0, 0.0], radius: 10.0}
        """
    )
    with pytest.raises(ValueError):
        load_config(bad_config)
