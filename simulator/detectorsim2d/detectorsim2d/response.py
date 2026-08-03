"""Calorimeter response: how much energy a particle leaves, and where.

A calorimeter measures a particle by destroying it -- the particle showers
into secondaries that ionize the active layers, and the detector sums that up.
This module is the toy model of that, with two ingredients:

**Longitudinally** (across layers), a shower's energy is split by a fixed set
of ``layer_fractions``, largest in the first layer and decreasing outward.
Real showers build up to a maximum a few radiation lengths in and then decay;
with only 3 (ECAL) / 2 (HCAL) sampling layers, a monotonically decreasing
profile is the honest simplification.

**Laterally** (across cells within a layer), the energy is spread as a
Gaussian in azimuth about the particle's own impact point, widening with
depth. The per-cell share is the *exact integral* of that Gaussian across the
cell's angular edges (an erf difference), not the density sampled at the cell
center -- so a shower landing on a cell boundary splits correctly, and the
total is conserved to the last digit regardless of how the cells happen to
line up. That matters here specifically: the ECAL's middle layer is staggered
by half a cell, so a shower centered on a boundary in layers 0 and 2 is
centered mid-cell in layer 1, and only an integral gets both right.

Three responses, selected by :mod:`detectorsim2d.species`'s ``interaction``:

- ``em`` -> :func:`shower` in the ECAL; nothing beyond.
- ``hadron`` -> :func:`mip` in the ECAL if charged, :func:`shower` in the HCAL.
- ``muon`` -> :func:`mip` everywhere; the muon system records position hits,
  not energy, so it is handled by the ordinary hit-finding pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd
from detector2d.calorimeter import CaloRing
from detector2d.propagate import SegmentedTrajectory, first_intersection_path

from . import species as species_module
from .edm import DEPOSITS_COLUMNS

_SQRT2 = math.sqrt(2.0)

#: How many sigma of the lateral Gaussian to spread energy over. Beyond ~3
#: sigma the per-cell share is below a per-mille of the shower and only adds
#: rows; the truncated tail is renormalized back in, so nothing is lost.
N_SIGMA = 3.0


@dataclass
class ShowerProfile:
    """Longitudinal and lateral shape of one shower type.

    ``layer_fractions`` is the share of the particle's energy deposited in
    each successive layer (normalized on use, so ``[3, 2, 1]`` and
    ``[0.5, 0.33, 0.17]`` mean the same thing). ``sigma_cells`` is the lateral
    Gaussian width in the *first* layer, measured in cells of that layer;
    ``sigma_growth`` widens it by that fraction per layer, since a shower
    spreads as it develops.
    """

    layer_fractions: tuple[float, ...] = (0.6, 0.3, 0.1)
    sigma_cells: float = 1.5
    sigma_growth: float = 0.5

    def fraction(self, index: int) -> float:
        total = sum(self.layer_fractions)
        if index >= len(self.layer_fractions) or total <= 0.0:
            return 0.0
        return self.layer_fractions[index] / total

    def sigma_phi(self, ring: CaloRing, index: int) -> float:
        return self.sigma_cells * (1.0 + self.sigma_growth * index) * ring.dphi


@dataclass
class ResponseConfig:
    """The full calorimeter response model.

    ``mip_energy`` is the absolute energy a minimum-ionizing particle leaves
    in one sampling layer of the given system -- a fixed amount, independent
    of the particle's energy, which is exactly what makes a muon's flat
    MIP trail distinguishable from a shower.

    ``stochastic`` is the calorimeter resolution term ``sigma_E/E = a/sqrt(E)``
    applied per layer. It defaults to 0 (no smearing) so that results are
    exactly reproducible; the shipped config turns it on.
    """

    #: `sigma_cells` is in units of the *layer's own* cells, so the ECAL's
    #: 1.5 fine cells and the HCAL's 1.0 coarse cells still come out ~4x
    #: apart in absolute width -- a hadronic shower really is much broader
    #: than an EM one, which is why the HCAL is segmented more coarsely.
    em: ShowerProfile = field(default_factory=lambda: ShowerProfile((0.60, 0.28, 0.12), 1.5, 0.5))
    hadron: ShowerProfile = field(default_factory=lambda: ShowerProfile((0.65, 0.35), 1.0, 0.4))
    mip_energy: dict[str, float] = field(default_factory=lambda: {"ecal": 0.3, "hcal": 0.8})
    stochastic: dict[str, float] = field(default_factory=dict)
    em_system: str = "ecal"
    hadron_system: str = "hcal"


def calo_rings(layers, system: str) -> list[CaloRing]:
    """The :class:`CaloRing`\\ s of one calorimeter, ordered inside-out."""
    rings = [layer for layer in layers if isinstance(layer, CaloRing) and layer.system == system]
    return sorted(rings, key=lambda ring: ring.radius)


def _impact_phi(path: SegmentedTrajectory, ring: CaloRing) -> float | None:
    """Azimuth at which ``path`` enters ``ring``, or ``None`` if it never gets
    there. Taken from the real intersection, so a track that bent on its way
    in deposits where it actually arrived, not where it was aimed."""
    hit = first_intersection_path(path, ring)
    if hit is None:
        return None
    cx, cy = ring.center
    return math.atan2(hit.y - cy, hit.x - cx)


def _gaussian_cell_shares(ring: CaloRing, phi_center: float, sigma_phi: float) -> list[tuple[int, float]]:
    """``(cell_index, share)`` pairs spreading a unit of energy over ``ring``
    as a Gaussian of width ``sigma_phi`` about ``phi_center``.

    Each cell's share is the Gaussian's integral between that cell's angular
    edges; the shares are renormalized over the truncated window so they sum
    to exactly 1. Cell indices wrap, so a shower straddling the +-pi branch
    cut is split across cells 0 and n_phi-1 rather than being clipped.
    """
    if sigma_phi <= 0.0:
        return [(ring.cell_index(phi_center), 1.0)]

    center_index = ring.cell_index(phi_center)
    reach = max(1, int(math.ceil(N_SIGMA * sigma_phi / ring.dphi)))

    shares: list[tuple[int, float]] = []
    for offset in range(-reach, reach + 1):
        index = (center_index + offset) % ring.n_phi
        low, high = ring.cell_edges(center_index + offset)
        # measure the cell's edges relative to the shower center, unwrapped
        z_low = math.remainder(low - phi_center, 2 * math.pi) / (_SQRT2 * sigma_phi)
        z_high = z_low + (high - low) / (_SQRT2 * sigma_phi)
        weight = 0.5 * (math.erf(z_high) - math.erf(z_low))
        if weight > 0.0:
            shares.append((index, weight))

    total = sum(weight for _, weight in shares)
    if total <= 0.0:
        return [(center_index, 1.0)]
    return [(index, weight / total) for index, weight in shares]


def _deposit_rows(ring: CaloRing, shares, energy: float, particle) -> list[dict]:
    return [
        dict(
            event_id=particle["event_id"],
            particle_id=particle["particle_id"],
            system=ring.system,
            layer_id=ring.layer_id,
            cell_id=index,
            x=ring.cell_position(index)[0],
            y=ring.cell_position(index)[1],
            s_local=ring.cell_local_coord(index),
            energy=share * energy,
        )
        for index, share in shares
    ]


def _smear(energy: float, system: str, config: ResponseConfig, rng) -> float:
    """Apply the ``a/sqrt(E)`` stochastic resolution term, clipped at zero."""
    a = config.stochastic.get(system, 0.0)
    if a <= 0.0 or rng is None or energy <= 0.0:
        return energy
    return max(0.0, rng.normal(energy, a * math.sqrt(energy)))


def shower(
    particle,
    path: SegmentedTrajectory,
    rings: list[CaloRing],
    energy: float,
    profile: ShowerProfile,
    config: ResponseConfig,
    rng=None,
) -> list[dict]:
    """Deposit ``energy`` as a shower across ``rings`` -- decreasing
    longitudinal fractions, Gaussian laterally, centered on the particle's own
    impact point in each layer."""
    rows: list[dict] = []
    for index, ring in enumerate(rings):
        phi = _impact_phi(path, ring)
        if phi is None:
            break  # absorbed or stopped before reaching this layer
        layer_energy = _smear(energy * profile.fraction(index), ring.system, config, rng)
        if layer_energy <= 0.0:
            continue
        shares = _gaussian_cell_shares(ring, phi, profile.sigma_phi(ring, index))
        rows.extend(_deposit_rows(ring, shares, layer_energy, particle))
    return rows


def mip(
    particle,
    path: SegmentedTrajectory,
    rings: list[CaloRing],
    config: ResponseConfig,
    rng=None,
) -> list[dict]:
    """Deposit a minimum-ionizing trail: one cell per layer, at the impact
    point, carrying a fixed energy that does not scale with the particle's
    own energy."""
    rows: list[dict] = []
    for ring in rings:
        phi = _impact_phi(path, ring)
        if phi is None:
            break
        energy = _smear(config.mip_energy.get(ring.system, 0.0), ring.system, config, rng)
        if energy <= 0.0:
            continue
        rows.extend(_deposit_rows(ring, [(ring.cell_index(phi), 1.0)], energy, particle))
    return rows


def respond(particle, path: SegmentedTrajectory, layers, config: ResponseConfig, rng=None) -> list[dict]:
    """Calorimeter deposits for one particle, dispatched on its species.

    Returns ``[]`` for a particle with no species (a bare kinematic stub) --
    it still leaves tracker hits, it just has no calorimeter response.
    """
    species = species_module.for_row(particle)
    if species is None:
        return []

    ecal = calo_rings(layers, config.em_system)
    hcal = calo_rings(layers, config.hadron_system)
    energy = float(particle.get("energy", 0.0) or 0.0)

    if species.interaction == species_module.EM:
        return shower(particle, path, ecal, energy, config.em, config, rng)

    if species.interaction == species_module.HADRON:
        # a charged hadron ionizes its way through the ECAL; a neutral one
        # passes through it invisibly
        rows = mip(particle, path, ecal, config, rng) if species.is_charged else []
        return rows + shower(particle, path, hcal, energy, config.hadron, config, rng)

    if species.interaction == species_module.MUON:
        return mip(particle, path, ecal, config, rng) + mip(particle, path, hcal, config, rng)

    raise ValueError(f"unknown interaction {species.interaction!r} for species {species.name!r}")


def sum_cells(deposits: pd.DataFrame) -> pd.DataFrame:
    """Collapse the truth-level deposits table to what a reconstruction sees:
    one row per cell, with every particle's contribution summed and the
    particle identity dropped."""
    if not len(deposits):
        return deposits.drop(columns=["particle_id"], errors="ignore")
    keys = ["event_id", "system", "layer_id", "cell_id", "x", "y", "s_local"]
    return deposits.groupby(keys, as_index=False)["energy"].sum()


__all__ = [
    "DEPOSITS_COLUMNS",
    "ResponseConfig",
    "ShowerProfile",
    "calo_rings",
    "mip",
    "respond",
    "shower",
    "sum_cells",
]
