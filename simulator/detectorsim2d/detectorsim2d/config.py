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


#: The three gun topologies -- see `ParticleGunConfig.mode`.
GUN_MODES: tuple[str, ...] = ("standard", "jets", "anomaly")


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

    #: Event topology -- one of `GUN_MODES`:
    #:
    #: - ``"standard"`` (default): every particle's `phi0` is drawn
    #:   independently and uniformly from ``phi_min``/``phi_max`` -- the
    #:   original behaviour.
    #: - ``"jets"``: the same ``n_particles`` multiplicity, but each particle
    #:   is assigned to one of a handful of jet axes (see `jet_count_min`/
    #:   `jet_count_max`) instead of an independent direction, so particles
    #:   cluster into 2-4 collimated sprays.
    #: - ``"anomaly"``: samples exactly like ``"standard"``, then -- with
    #:   probability `anomaly_rate` per event -- overlays an injected,
    #:   unphysical topology: two high-energy calorimeter showers lined up
    #:   back-to-back, with a mu+ mu- pair emitted along the same axis. Meant
    #:   as a planted signal for exercising anomaly-detection algorithms, not
    #:   a physical process.
    mode: str = "standard"

    #: `jets` mode: the number of jet axes per event is drawn uniformly from
    #: [`jet_count_min`, `jet_count_max`].
    jet_count_min: int = 2
    jet_count_max: int = 4
    #: `jets` mode: angular (phi) spread of a jet's particles about its axis,
    #: in radians -- the jet "cone" width.
    jet_cone_sigma: float = 0.15
    #: `jets` mode: fraction of jets, per event, that are "b-jets" -- instead
    #: of starting at the primary vertex (`vertex_x`/`vertex_y`), the jet's
    #: particles all originate from a shared point displaced along the jet's
    #: own axis, standing in for an invisible parent (e.g. a B hadron) that
    #: flies some distance through the tracker before decaying into the
    #: visible jet. The rest of a b-jet's particles (species, pt, cone
    #: spread) are sampled exactly like any other jet's.
    b_jet_fraction: float = 0.15
    #: `jets` mode, b-jets only: the displaced vertex's flight length along
    #: the jet axis is drawn uniformly from [`b_jet_decay_length_min`,
    #: `b_jet_decay_length_max`] -- tune these relative to the detector's own
    #: tracker radius so the decay point lands "somewhere in the tracker
    #: volume" rather than past it.
    b_jet_decay_length_min: float = 0.0
    b_jet_decay_length_max: float = 20.0

    #: `anomaly` mode: probability, per event, that the anomalous cluster is
    #: injected on top of the standard particles.
    anomaly_rate: float = 0.3
    #: `anomaly` mode: species used for the two back-to-back calorimeter
    #: showers (must be an EM or hadron species; see `detectorsim2d.species`).
    anomaly_calo_species: str = "photon"
    #: `anomaly` mode: energy of the calorimeter showers / muons, each as a
    #: multiple of `pt_max` -- scaled off the gun's own range rather than an
    #: absolute number, so the anomaly reads as "high energetic" regardless
    #: of how a given config's `pt_min`/`pt_max` are tuned.
    anomaly_calo_scale: float = 2.0
    anomaly_muon_scale: float = 1.5
    #: `anomaly` mode: random jitter (radians) added to each of the four
    #: particles' `phi0` around the shared axis. 0.0 means exactly lined up.
    anomaly_axis_jitter: float = 0.0

    def __post_init__(self) -> None:
        if self.mode not in GUN_MODES:
            raise ValueError(f"unknown gun mode {self.mode!r}; known: {', '.join(GUN_MODES)}")

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
