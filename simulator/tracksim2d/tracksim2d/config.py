"""Configuration dataclasses for the 2D tracking simulation.

All fields default to reasonable values, so ``SimConfig()`` with an empty
``layers`` list and no YAML file still works (it just produces particles with
no hits). Follows the same load/merge pattern as
`sensor/sim/config.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from detector2d.geometry import CircleLayer, LineLayer

Layer = LineLayer | CircleLayer


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


def load_config(path: str | Path | None) -> SimConfig:
    """Load a SimConfig from a YAML file, falling back to defaults for
    anything not present. ``path=None`` returns pure defaults."""
    if path is None:
        return SimConfig()

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    kwargs: dict[str, Any] = {}
    if "layers" in raw:
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
