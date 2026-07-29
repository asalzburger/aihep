"""Declarative detector-layout description: parse a plain dict (typically
loaded from YAML by a downstream package) into the flat list of
:class:`~detector2d.geometry.LineLayer`/:class:`~detector2d.geometry.CircleLayer`
objects that :mod:`detector2d.intersect` (and simulation code built on top of
it, e.g. `tracksim2d`) consume.

Two mutually exclusive ways to describe a layout, both handled by
:func:`build_layers_from_raw`:

- ``layers:`` -- a flat, hand-listed list of line/circle layer specs
  (:func:`parse_layer`).
- ``detector:`` -- a higher-level cylindrical/barrel spec
  (:class:`DetectorConfig`) that expands into the same flat layer list via
  :mod:`detector2d.barrel` (:func:`build_detector_layers`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any

from .barrel import build_barrel_circle, build_barrel_modules
from .geometry import CircleLayer, LineLayer

Layer = LineLayer | CircleLayer


@dataclass
class ModuleTypeConfig:
    """Module geometry shared by every layer of a given ``kind`` (e.g. all
    "precision" layers use the same module size/tilt/overlap) -- see
    :func:`detector2d.barrel.build_barrel_modules`."""

    half_length: float
    tilt_deg: float
    overlap_fraction: float
    pitch: float | None = None


@dataclass
class DetectorLayerConfig:
    layer_id: int
    radius: float
    kind: str


@dataclass
class DetectorConfig:
    """A high-level, declarative cylindrical-detector spec that expands into
    a flat layer list -- see :func:`build_detector_layers`. Alternative to
    (and mutually exclusive with) hand-listing layers directly."""

    mode: str = "simplified"  # "simplified" (bare CircleLayer) or "detailed" (tilted modules)
    layers: list[DetectorLayerConfig] = field(default_factory=list)
    module_types: dict[str, ModuleTypeConfig] = field(default_factory=dict)


def _merge_dataclass(cls: type, data: dict[str, Any] | None):
    """Build a flat dataclass (no nested dataclass fields) from a dict,
    keeping the dataclass default for any key not present in ``data``."""
    data = data or {}
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    return cls(**kwargs)


def parse_layer(spec: dict[str, Any]) -> Layer:
    kind = spec["kind"]
    if kind == "line":
        return LineLayer(
            layer_id=spec["layer_id"], p1=tuple(spec["p1"]), p2=tuple(spec["p2"]), pitch=spec.get("pitch")
        )
    if kind == "circle":
        return CircleLayer(
            layer_id=spec["layer_id"],
            center=tuple(spec["center"]),
            radius=spec["radius"],
            pitch=spec.get("pitch"),
        )
    raise ValueError(f"Unknown layer kind: {kind!r}")


def parse_detector_config(raw: dict[str, Any]) -> DetectorConfig:
    layers = [
        DetectorLayerConfig(layer_id=spec["layer_id"], radius=spec["radius"], kind=spec["kind"])
        for spec in raw.get("layers", [])
    ]
    module_types = {
        name: _merge_dataclass(ModuleTypeConfig, spec) for name, spec in raw.get("module_types", {}).items()
    }
    return DetectorConfig(mode=raw.get("mode", "simplified"), layers=layers, module_types=module_types)


def build_detector_layers(detector: DetectorConfig) -> list[Layer]:
    """Expand a declarative :class:`DetectorConfig` into the flat list of
    ``detector2d`` layer objects: one ``CircleLayer`` per layer in
    ``mode="simplified"``, or a ring of tilted ``LineLayer`` modules per
    layer (see :mod:`detector2d.barrel`) in ``mode="detailed"`` -- all
    modules for one physical layer share its ``layer_id``, which is what
    lets a track crossing an overlap between two neighboring modules show
    up as two hits on that one layer."""
    if detector.mode not in ("simplified", "detailed"):
        raise ValueError(f"Unknown detector mode: {detector.mode!r}")

    layers: list[Layer] = []
    for layer_spec in detector.layers:
        module = detector.module_types[layer_spec.kind]
        if detector.mode == "simplified":
            layers.append(build_barrel_circle(layer_spec.layer_id, layer_spec.radius, pitch=module.pitch))
        else:
            layers.extend(
                build_barrel_modules(
                    layer_id=layer_spec.layer_id,
                    radius=layer_spec.radius,
                    half_length=module.half_length,
                    tilt=math.radians(module.tilt_deg),
                    overlap_fraction=module.overlap_fraction,
                    pitch=module.pitch,
                )
            )
    return layers


def build_layers_from_raw(raw: dict[str, Any]) -> list[Layer]:
    """Build the flat layer list from a config dict's ``detector:``/
    ``layers:`` keys (mutually exclusive -- pick one). ``raw`` is the
    top-level config dict (e.g. parsed from YAML by the caller), not just
    the ``detector:`` sub-dict. Returns ``[]`` if neither key is present."""
    if "detector" in raw and "layers" in raw:
        raise ValueError(
            "config has both 'detector' and 'layers' -- these are mutually exclusive ways to "
            "specify the detector layout, pick one"
        )
    if "detector" in raw:
        return build_detector_layers(parse_detector_config(raw["detector"]))
    if "layers" in raw:
        return [parse_layer(spec) for spec in raw["layers"]]
    return []
