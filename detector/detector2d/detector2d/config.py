"""Declarative detector-layout description: parse a plain dict (typically
loaded from YAML by a downstream package) into the flat list of
:class:`~detector2d.geometry.LineLayer`/:class:`~detector2d.geometry.CircleLayer`
objects that :mod:`detector2d.intersect` (and simulation code built on top of
it, e.g. `detectorsim2d`) consume.

The *tracker* is described in one of two mutually exclusive ways:

- ``layers:`` -- a flat, hand-listed list of line/circle layer specs
  (:func:`parse_layer`).
- ``detector:`` -- a higher-level cylindrical/barrel spec
  (:class:`DetectorConfig`) that expands into the same flat layer list via
  :mod:`detector2d.barrel` (:func:`build_detector_layers`).

Two further, independent and optional blocks describe what sits outside it,
and simply append to whichever of the above produced the tracker:

- ``calorimeter:`` -- named calorimeter stacks (``ecal``, ``hcal``, ...),
  expanded via :mod:`detector2d.calorimeter` (:func:`build_calorimeter_layers`).
- ``muon:`` -- polygonal triplet stations, expanded via
  :mod:`detector2d.polygon` (:func:`build_muon_layers`).

:func:`build_layers_from_raw` assembles all of them into one flat list. A
config with none of these keys yields ``[]``; a config with only
``layers:``/``detector:`` yields exactly what it always did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any

from .barrel import build_barrel_circle, build_barrel_modules
from .calorimeter import CaloRing, CaloStackConfig, build_calo_stack_from_config
from .field import DEFAULT_K, FieldRegion, FieldRegions
from .geometry import CircleLayer, LineLayer
from .polygon import build_muon_system

Layer = LineLayer | CircleLayer | CaloRing


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
    system = spec.get("system", "tracker")
    if kind == "line":
        return LineLayer(
            layer_id=spec["layer_id"],
            p1=tuple(spec["p1"]),
            p2=tuple(spec["p2"]),
            pitch=spec.get("pitch"),
            system=system,
        )
    if kind == "circle":
        return CircleLayer(
            layer_id=spec["layer_id"],
            center=tuple(spec["center"]),
            radius=spec["radius"],
            pitch=spec.get("pitch"),
            system=system,
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


@dataclass
class MuonConfig:
    """A polygonal muon spectrometer: ``n_stations`` equally spaced stations,
    each an ``n_sides``-gon whose every side is an ``n_planes`` triplet -- see
    :func:`detector2d.polygon.build_muon_system`."""

    layer_id_base: int = 300
    apothem_inner: float = 520.0
    station_spacing: float = 100.0
    n_stations: int = 3
    n_planes: int = 3
    n_sides: int = 8
    triplet_gap: float = 8.0
    phi_offset_deg: float = 0.0
    pitch: float | None = None
    station_id_step: int = 10
    system: str = "muon"


def parse_calorimeter_config(raw: dict[str, Any]) -> list[CaloStackConfig]:
    """Parse a ``calorimeter:`` block -- a mapping of stack name (``ecal``,
    ``hcal``, ...) to its :class:`~detector2d.calorimeter.CaloStackConfig`
    fields. The key doubles as the stack's ``system`` tag unless the spec
    overrides it."""
    stacks = []
    for name, spec in raw.items():
        merged = _merge_dataclass(CaloStackConfig, {"system": name, **(spec or {})})
        stacks.append(merged)
    return stacks


def build_calorimeter_layers(stacks: list[CaloStackConfig]) -> list[Layer]:
    layers: list[Layer] = []
    for stack in stacks:
        layers.extend(build_calo_stack_from_config(stack))
    return layers


def parse_muon_config(raw: dict[str, Any]) -> MuonConfig:
    return _merge_dataclass(MuonConfig, raw)


def build_muon_layers(muon: MuonConfig) -> list[Layer]:
    return list(
        build_muon_system(
            layer_id_base=muon.layer_id_base,
            apothem_inner=muon.apothem_inner,
            station_spacing=muon.station_spacing,
            n_stations=muon.n_stations,
            n_planes=muon.n_planes,
            n_sides=muon.n_sides,
            triplet_gap=muon.triplet_gap,
            phi_offset=math.radians(muon.phi_offset_deg),
            pitch=muon.pitch,
            system=muon.system,
            station_id_step=muon.station_id_step,
        )
    )


def parse_field_regions(raw: dict[str, Any] | None) -> FieldRegions:
    """Parse a ``field:`` block into :class:`~detector2d.field.FieldRegions`.

    Two forms. The piecewise one, inside-out, the last entry's ``r_max``
    omitted (or null) for "everything beyond"::

        field:
          k: 0.2998
          regions:
            - {r_max: 210.0, bz:  2.0}   # tracker: strong
            - {r_max: 480.0, bz:  0.0}   # calorimeters: no field
            - {bz: -1.0}                 # muon system: half, reversed

    and the original scalar form, ``field: {bz: 1.0}``, which still means one
    constant field everywhere -- it parses to a single unbounded region.
    """
    raw = raw or {}
    k = raw.get("k", DEFAULT_K)
    if "regions" in raw:
        regions = tuple(
            FieldRegion(r_max=spec.get("r_max"), bz=spec.get("bz", 0.0)) for spec in raw["regions"]
        )
        return FieldRegions(regions=regions, k=k)
    return FieldRegions.constant(bz=raw.get("bz", 0.0), k=k)


def build_layers_from_raw(raw: dict[str, Any]) -> list[Layer]:
    """Build the flat layer list from a config dict's ``detector:``/``layers:``
    (mutually exclusive -- pick one), plus the optional ``calorimeter:`` and
    ``muon:`` blocks, in that inside-out order. ``raw`` is the top-level
    config dict (e.g. parsed from YAML by the caller), not any one sub-dict.
    Returns ``[]`` if none of the keys is present."""
    if "detector" in raw and "layers" in raw:
        raise ValueError(
            "config has both 'detector' and 'layers' -- these are mutually exclusive ways to "
            "specify the detector layout, pick one"
        )

    layers: list[Layer] = []
    if "detector" in raw:
        layers.extend(build_detector_layers(parse_detector_config(raw["detector"])))
    elif "layers" in raw:
        layers.extend(parse_layer(spec) for spec in raw["layers"])
    if "calorimeter" in raw:
        layers.extend(build_calorimeter_layers(parse_calorimeter_config(raw["calorimeter"])))
    if "muon" in raw:
        layers.extend(build_muon_layers(parse_muon_config(raw["muon"])))
    return layers
