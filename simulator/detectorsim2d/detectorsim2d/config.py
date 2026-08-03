"""Configuration dataclasses for the 2D detector simulation.

All fields default to reasonable values, so ``SimConfig()`` with an empty
``layers`` list and no YAML file still works (it just produces particles with
no hits). Follows the same load/merge pattern as `sensor/sim/config.py`.

Detector layout (the ``layers:``/``detector:``/``calorimeter:``/``muon:`` YAML
keys) is not this package's concern -- it's parsed by :mod:`detector2d.config`
(see :func:`detector2d.config.build_layers_from_raw`); this module owns the
simulation-side config: the field, the particle gun, and the calorimeter
response model.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml
from detector2d.config import Layer, build_layers_from_raw, parse_field_regions
from detector2d.field import FieldRegions

from .response import ResponseConfig, ShowerProfile


@dataclass
class FieldConfig:
    """The innermost (or only) field, and the momentum-to-radius constant.

    With a piecewise field map (``SimConfig.field_regions``) ``bz`` is the
    *innermost* region's field -- the one that sets each particle's nominal
    ``radius`` in the particles table.
    """

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
    #: Species names to draw from (see :mod:`detectorsim2d.species`), with
    #: optional relative weights. When empty the gun falls back to the
    #: original species-free behaviour: a bare charged stub drawn from
    #: ``charges``, which bends and leaves hits but has no calorimeter
    #: response.
    species: tuple[str, ...] = ()
    species_weights: tuple[float, ...] = ()
    charges: tuple[float, ...] = (-1.0, 1.0)
    pt_min: float = 1.0
    pt_max: float = 10.0

    @property
    def species_probabilities(self) -> list[float] | None:
        """``species_weights`` normalized for ``rng.choice``, or ``None`` for
        a uniform draw."""
        if not self.species_weights:
            return None
        if len(self.species_weights) != len(self.species):
            raise ValueError(
                f"species_weights has {len(self.species_weights)} entries but species has "
                f"{len(self.species)} -- they must line up one to one"
            )
        total = sum(self.species_weights)
        if total <= 0:
            raise ValueError("species_weights must sum to something positive")
        return [w / total for w in self.species_weights]


@dataclass
class SimConfig:
    layers: list[Layer] = field(default_factory=list)
    magnetic_field: FieldConfig = field(default_factory=FieldConfig)
    #: Piecewise-radial field map. ``None`` means the single constant
    #: ``magnetic_field.bz`` everywhere -- the pre-calorimeter behaviour, and
    #: what every ``field: {bz: ...}`` config still gets.
    field_regions: FieldRegions | None = None
    gun: ParticleGunConfig = field(default_factory=ParticleGunConfig)
    #: Calorimeter response model. ``None`` disables calorimetry entirely
    #: (no deposits table), which is what a tracker-only config wants.
    response: ResponseConfig | None = None
    n_events: int = 1
    seed: int | None = None
    #: Outer radius of the tracker volume, centered at the origin. Passed to
    #: `detectorsim2d.simulate.propagate_particles`, which stops propagating a
    #: particle once it first exits this radius -- past that point a curved
    #: arc would otherwise loop back inward, which isn't physically
    #: meaningful once the particle has left the tracker. `detectorsim2d.vis`
    #: applies the same cutoff to the drawn curve itself. ``None`` disables
    #: the cap. In a full detector this is superseded by `world_radius`.
    tracker_boundary: float | None = None
    #: Outer radius of the whole detector: propagation stops there.
    world_radius: float | None = None
    #: Hard cap on a particle's path length. This is what terminates a
    #: low-`pt` particle whose bend radius is too small to ever leave the
    #: tracker -- without it, it circles forever.
    max_path_length: float | None = None


def _merge_dataclass(cls: type, data: dict[str, Any] | None):
    """Build a flat dataclass (no nested dataclass fields) from a dict,
    keeping the dataclass default for any key not present in ``data``."""
    data = data or {}
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    return cls(**kwargs)


def parse_shower_profile(raw: dict[str, Any] | None, default: ShowerProfile) -> ShowerProfile:
    raw = raw or {}
    return ShowerProfile(
        layer_fractions=tuple(raw.get("layer_fractions", default.layer_fractions)),
        sigma_cells=raw.get("sigma_cells", default.sigma_cells),
        sigma_growth=raw.get("sigma_growth", default.sigma_growth),
    )


def parse_response(raw: dict[str, Any] | None) -> ResponseConfig:
    """Parse a ``response:`` block into a :class:`ResponseConfig`::

        response:
          em:     {layer_fractions: [0.60, 0.28, 0.12], sigma_cells: 1.5, sigma_growth: 0.5}
          hadron: {layer_fractions: [0.65, 0.35],       sigma_cells: 2.5, sigma_growth: 0.4}
          mip_energy: {ecal: 0.3, hcal: 0.8}
          stochastic: {ecal: 0.10, hcal: 0.50}
    """
    raw = raw or {}
    defaults = ResponseConfig()
    return ResponseConfig(
        em=parse_shower_profile(raw.get("em"), defaults.em),
        hadron=parse_shower_profile(raw.get("hadron"), defaults.hadron),
        mip_energy=dict(raw.get("mip_energy", defaults.mip_energy)),
        stochastic=dict(raw.get("stochastic", defaults.stochastic)),
        em_system=raw.get("em_system", defaults.em_system),
        hadron_system=raw.get("hadron_system", defaults.hadron_system),
    )


def load_config(path: str | Path | None) -> SimConfig:
    """Load a SimConfig from a YAML file, falling back to defaults for
    anything not present. ``path=None`` returns pure defaults."""
    if path is None:
        return SimConfig()

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    kwargs: dict[str, Any] = {"layers": build_layers_from_raw(raw)}

    if "field" in raw:
        regions = parse_field_regions(raw["field"])
        # the innermost region's bz is what sets each particle's nominal
        # radius; a scalar `field: {bz: ...}` stays a plain constant field
        kwargs["magnetic_field"] = FieldConfig(bz=regions.regions[0].bz, k=regions.k)
        if "regions" in raw["field"]:
            kwargs["field_regions"] = regions

    if "gun" in raw:
        gun_raw = dict(raw["gun"])
        for key in ("charges", "species", "species_weights"):
            if key in gun_raw:
                gun_raw[key] = tuple(gun_raw[key])
        kwargs["gun"] = _merge_dataclass(ParticleGunConfig, gun_raw)

    if "response" in raw:
        kwargs["response"] = parse_response(raw["response"])

    for key in ("n_events", "seed", "tracker_boundary", "world_radius", "max_path_length"):
        if key in raw:
            kwargs[key] = raw[key]

    return SimConfig(**kwargs)
