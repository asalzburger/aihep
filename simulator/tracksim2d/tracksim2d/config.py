"""Configuration dataclasses for the 2D tracking simulation.

All fields default to reasonable values, so ``SimConfig()`` with an empty
``layers`` list and no YAML file still works (it just produces particles with
no hits). Follows the same load/merge pattern as
`sensor/sim/config.py`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from detector2d.barrel import build_barrel_circle, build_barrel_modules
from detector2d.geometry import CircleLayer, LineLayer

Layer = LineLayer | CircleLayer


@dataclass
class ModuleTypeConfig:
    """Module geometry shared by every layer of a given ``kind`` (e.g. all
    "precision" layers use the same module size/tilt/overlap) -- see
    detector2d.barrel.build_barrel_modules."""

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
    ``SimConfig.layers`` -- see :func:`build_detector_layers`. Alternative to
    (and mutually exclusive with) hand-listing ``layers:`` directly."""

    mode: str = "simplified"  # "simplified" (bare CircleLayer) or "detailed" (tilted modules)
    layers: list[DetectorLayerConfig] = field(default_factory=list)
    module_types: dict[str, ModuleTypeConfig] = field(default_factory=dict)


@dataclass
class FieldConfig:
    bz: float = 0.0  # constant field out of the 2D plane
    k: float = 0.2998  # R[len] = pt / (k * |q| * bz); see detector2d.field.signed_radius


@dataclass
class ParticleGunConfig:
    n_particles: int = 1
    vertex_x: float = 0.0
    vertex_y: float = 0.0
    vertex_spread_x: float = 0.0  # uniform +/- spread around vertex_x
    vertex_spread_y: float = 0.0
    phi_min: float = -0.3
    phi_max: float = 0.3
    charges: tuple[float, ...] = (-1.0, 1.0)
    pt_min: float = 1.0
    pt_max: float = 10.0


@dataclass
class SimConfig:
    layers: list[Layer] = field(default_factory=list)
    magnetic_field: FieldConfig = field(default_factory=FieldConfig)
    gun: ParticleGunConfig = field(default_factory=ParticleGunConfig)
    n_events: int = 1
    seed: int | None = None


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
    ``detector2d`` layer objects ``SimConfig.layers`` expects: one
    ``CircleLayer`` per layer in ``mode="simplified"``, or a ring of tilted
    ``LineLayer`` modules per layer (see :mod:`detector2d.barrel`) in
    ``mode="detailed"`` -- all modules for one physical layer share its
    ``layer_id``, which is what lets a track crossing an overlap between two
    neighboring modules show up as two hits on that one layer."""
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


def load_config(path: str | Path | None) -> SimConfig:
    """Load a SimConfig from a YAML file, falling back to defaults for
    anything not present. ``path=None`` returns pure defaults."""
    if path is None:
        return SimConfig()

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    kwargs: dict[str, Any] = {}
    if "detector" in raw and "layers" in raw:
        raise ValueError(
            "config has both 'detector' and 'layers' -- these are mutually exclusive ways to "
            "specify the detector layout, pick one"
        )
    if "detector" in raw:
        kwargs["layers"] = build_detector_layers(parse_detector_config(raw["detector"]))
    elif "layers" in raw:
        kwargs["layers"] = [parse_layer(spec) for spec in raw["layers"]]
    if "field" in raw:
        kwargs["magnetic_field"] = _merge_dataclass(FieldConfig, raw["field"])
    if "gun" in raw:
        gun_raw = dict(raw["gun"])
        if "charges" in gun_raw:
            gun_raw["charges"] = tuple(gun_raw["charges"])
        kwargs["gun"] = _merge_dataclass(ParticleGunConfig, gun_raw)
    for key in ("n_events", "seed"):
        if key in raw:
            kwargs[key] = raw[key]

    return SimConfig(**kwargs)
