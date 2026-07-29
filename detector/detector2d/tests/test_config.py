import pytest

from detector2d.config import (
    DetectorConfig,
    build_detector_layers,
    build_layers_from_raw,
    parse_detector_config,
    parse_layer,
)
from detector2d.geometry import CircleLayer, LineLayer

BARREL6_RAW = {
    "mode": "detailed",
    "layers": [
        {"layer_id": 0, "radius": 29.0, "kind": "precision"},
        {"layer_id": 1, "radius": 48.0, "kind": "precision"},
        {"layer_id": 2, "radius": 68.0, "kind": "precision"},
        {"layer_id": 3, "radius": 100.0, "kind": "outer"},
        {"layer_id": 4, "radius": 140.0, "kind": "outer"},
        {"layer_id": 5, "radius": 200.0, "kind": "outer"},
    ],
    "module_types": {
        "precision": {"half_length": 4.0, "tilt_deg": 10.0, "overlap_fraction": 0.15, "pitch": 0.1},
        "outer": {"half_length": 8.0, "tilt_deg": 8.0, "overlap_fraction": 0.10, "pitch": 0.5},
    },
}


def test_parse_layer_line_and_circle():
    line = parse_layer({"kind": "line", "layer_id": 0, "p1": [10.0, -50.0], "p2": [10.0, 50.0], "pitch": 1.0})
    assert line == LineLayer(layer_id=0, p1=(10.0, -50.0), p2=(10.0, 50.0), pitch=1.0)

    circle = parse_layer({"kind": "circle", "layer_id": 5, "center": [0.0, 0.0], "radius": 5.0, "pitch": 0.5})
    assert circle == CircleLayer(layer_id=5, center=(0.0, 0.0), radius=5.0, pitch=0.5)


def test_parse_layer_rejects_unknown_kind():
    with pytest.raises(ValueError):
        parse_layer({"kind": "bogus", "layer_id": 0})


def test_parse_detector_config_and_build_simplified_is_bare_circles():
    raw = dict(BARREL6_RAW, mode="simplified")
    detector = parse_detector_config(raw)
    layers = build_detector_layers(detector)
    assert len(layers) == 6
    assert all(isinstance(layer, CircleLayer) for layer in layers)
    assert [layer.radius for layer in layers] == [29.0, 48.0, 68.0, 100.0, 140.0, 200.0]


def test_build_detector_layers_detailed_builds_tilted_module_rings():
    detector = parse_detector_config(BARREL6_RAW)
    layers = build_detector_layers(detector)
    assert all(isinstance(layer, LineLayer) for layer in layers)
    layer_ids = {layer.layer_id for layer in layers}
    assert layer_ids == {0, 1, 2, 3, 4, 5}
    per_layer_counts = {lid: sum(1 for layer in layers if layer.layer_id == lid) for lid in layer_ids}
    assert all(count > 1 for count in per_layer_counts.values())
    assert all(layer.pitch == 0.1 for layer in layers if layer.layer_id in (0, 1, 2))
    assert all(layer.pitch == 0.5 for layer in layers if layer.layer_id in (3, 4, 5))


def test_build_detector_layers_rejects_unknown_mode():
    detector = DetectorConfig(mode="bogus")
    with pytest.raises(ValueError):
        build_detector_layers(detector)


def test_build_layers_from_raw_dispatches_on_detector_or_layers_key():
    assert build_layers_from_raw({}) == []

    flat = build_layers_from_raw({"layers": [{"kind": "circle", "layer_id": 0, "center": [0.0, 0.0], "radius": 10.0}]})
    assert flat == [CircleLayer(layer_id=0, center=(0.0, 0.0), radius=10.0)]

    detector_raw = {"detector": dict(BARREL6_RAW, mode="simplified")}
    from_detector = build_layers_from_raw(detector_raw)
    assert len(from_detector) == 6
    assert all(isinstance(layer, CircleLayer) for layer in from_detector)


def test_build_layers_from_raw_rejects_both_detector_and_layers():
    raw = {
        "detector": dict(BARREL6_RAW, mode="simplified"),
        "layers": [{"kind": "circle", "layer_id": 0, "center": [0.0, 0.0], "radius": 10.0}],
    }
    with pytest.raises(ValueError):
        build_layers_from_raw(raw)
