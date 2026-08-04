"""Track and cluster reconstruction: turn `detectorsim2d` truth into the
smeared, detector-resolution-limited quantities a real reconstruction would
actually measure.

**Tracks.** Every *charged* particle gets a track: its impact parameter
(`d0`, from `detector2d.geometry.Trajectory.d0`), initial direction
(`phi0`), and transverse momentum (`pt`) are each smeared by an independent
Gaussian whose width shrinks with `pt` (see :func:`resolution`) -- a higher
-energy track leaves a straighter, more precisely measured trace. **Muons
are not special-cased here**: a muon is charged and ionizes its way through
the tracker exactly like any other charged particle, so it becomes a track
the same way a pion does. What *does* treat it differently is
:func:`reconstruct_clusters`, next.

**Clusters.** Every particle whose species showers (EM or hadronic --
see `detectorsim2d.species`) gets a cluster: its total truth calorimeter
deposit (summed across every cell it hit, in whichever system(s) -- a
charged hadron's ECAL MIP trail and HCAL shower both count, the honest
simplification for a first reconstruction pass) is smeared by one more
Gaussian, same shrinks-with-energy resolution law. A muon's own MIP trail is
*excluded* -- it is reconstructed as a track, not a cluster, which is what
"muons act like tracks" means in practice.

Both tables carry `jet_id`/`is_b_jet` and `species` straight through from
the particles table as truth context for validation -- see the module
docstring of :mod:`detectorreco2d.edm`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from detectorsim2d import species as species_module
from detectorsim2d.simulate import trajectory_for_row

from .config import RecoConfig
from .edm import CLUSTERS_COLUMNS, TRACKS_COLUMNS

#: Floor a smeared pt is clipped to -- a Gaussian can wander through zero,
#: which no real track (a curved-or-straight ray with a direction) can have.
_MIN_PT = 1.0e-6


def resolution(a: float, b: float, x: float) -> float:
    """Gaussian sigma for a quantity measured alongside scale variable ``x``
    (a track's own `pt`, or a cluster's own `energy`): ``a + b / x``, shrinking
    toward the asymptotic floor ``a`` as ``x`` grows -- the standard "better
    resolution at higher energy" detector behaviour. Falls back to the floor
    ``a`` alone for ``x <= 0`` (guards the division, and there is nothing
    meaningful to divide by for a non-positive energy/pt anyway)."""
    if x <= 0:
        return a
    return a + b / x


def _with_jet_truth_defaults(particles: pd.DataFrame) -> pd.DataFrame:
    """`jet_id`/`is_b_jet`, filled in for particles tables that predate jets
    mode (missing the columns entirely) or that simply weren't produced by
    it (`NaN`/`jet_id=-1`, `is_b_jet=False`, same as `detectorsim2d` itself
    uses for non-jets particles)."""
    particles = particles.copy()
    if "jet_id" not in particles.columns:
        particles["jet_id"] = -1
    if "is_b_jet" not in particles.columns:
        particles["is_b_jet"] = False
    particles["jet_id"] = particles["jet_id"].fillna(-1).astype(int)
    particles["is_b_jet"] = particles["is_b_jet"].fillna(False).astype(bool)
    return particles


def _is_showering(row) -> bool:
    """Whether this particle's species leaves a calorimeter shower at all --
    EM or hadronic; `None` (a bare, species-free stub) and muons (MIP only)
    are both excluded."""
    species = species_module.for_row(row)
    return species is not None and species.interaction in (species_module.EM, species_module.HADRON)


def reconstruct_tracks(
    particles: pd.DataFrame, config: RecoConfig, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    """Smear every charged particle's `(d0, phi0, pt)` into a track -- see
    the module docstring for why muons need no special handling here."""
    if rng is None:
        rng = np.random.default_rng(config.seed)

    particles = _with_jet_truth_defaults(particles)
    charged = particles[particles["charge"] != 0]

    rows = []
    for _, particle in charged.iterrows():
        pt_true = float(particle["energy"])
        d0_true = trajectory_for_row(particle).d0
        phi0_true = float(particle["phi0"])

        pt = max(_MIN_PT, rng.normal(pt_true, resolution(config.track.pt.a, config.track.pt.b, pt_true)))
        d0 = rng.normal(d0_true, resolution(config.track.d0.a, config.track.d0.b, pt_true))
        phi0 = rng.normal(phi0_true, resolution(config.track.phi0.a, config.track.phi0.b, pt_true))

        rows.append(
            dict(
                event_id=particle["event_id"],
                particle_id=particle["particle_id"],
                jet_id=int(particle["jet_id"]),
                is_b_jet=bool(particle["is_b_jet"]),
                species=particle.get("species", np.nan),
                charge=particle["charge"],
                d0_true=d0_true,
                d0=d0,
                phi0_true=phi0_true,
                phi0=phi0,
                pt_true=pt_true,
                pt=pt,
            )
        )
    return pd.DataFrame(rows, columns=TRACKS_COLUMNS)


def reconstruct_clusters(
    particles: pd.DataFrame,
    deposits: pd.DataFrame,
    config: RecoConfig,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Smear each showering particle's total truth calorimeter deposit into
    a cluster -- see the module docstring for what's excluded and why."""
    if rng is None:
        rng = np.random.default_rng(config.seed)

    particles = _with_jet_truth_defaults(particles)
    showering = particles[particles.apply(_is_showering, axis=1)]
    if not len(deposits) or not len(showering):
        return pd.DataFrame(columns=CLUSTERS_COLUMNS)

    totals = deposits.groupby(["event_id", "particle_id"], as_index=False)["energy"].sum()
    info = showering.set_index(["event_id", "particle_id"])

    rows = []
    for _, total in totals.iterrows():
        key = (total["event_id"], total["particle_id"])
        if key not in info.index:
            continue  # a muon's MIP deposits, or another non-showering particle
        particle = info.loc[key]
        energy_true = float(total["energy"])
        sigma = resolution(config.cluster.energy.a, config.cluster.energy.b, energy_true)
        energy = max(0.0, rng.normal(energy_true, sigma))
        rows.append(
            dict(
                event_id=key[0],
                particle_id=key[1],
                jet_id=int(particle["jet_id"]),
                is_b_jet=bool(particle["is_b_jet"]),
                species=particle.get("species", np.nan),
                energy_true=energy_true,
                energy=energy,
            )
        )
    return pd.DataFrame(rows, columns=CLUSTERS_COLUMNS)


def reconstruct(
    particles: pd.DataFrame,
    deposits: pd.DataFrame,
    config: RecoConfig,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """`(tracks, clusters)` for a full run -- the two-call convenience
    entry point, mirroring `detectorsim2d.simulate.simulate_events`."""
    if rng is None:
        rng = np.random.default_rng(config.seed)
    return reconstruct_tracks(particles, config, rng), reconstruct_clusters(particles, deposits, config, rng)
