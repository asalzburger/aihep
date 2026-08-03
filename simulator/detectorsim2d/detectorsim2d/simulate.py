"""Event generation: sample particles (or take a given ground truth) and
propagate them through a detector to produce hits and calorimeter deposits.

Split on purpose:

- :func:`propagate_particles` turns *any* particles table into
  ``(hits, deposits)`` by propagating each row through the layers. It doesn't
  care where the particles came from -- a random gun, or (as in the Denby
  recreation) parameters fit straight out of a reference picture.
  :func:`hits_for_particles` is the hits-only view of it, unchanged in
  signature and behaviour from before the calorimeter existed.
- :func:`simulate_events` is the random-gun path: sample particles from a
  :class:`~detectorsim2d.config.ParticleGunConfig`, then propagate them.

**Where a particle stops is what distinguishes it.** Rather than a special
mechanism, absorption reuses the boundary cutoff that already existed for the
tracker: each particle gets a stopping radius from its species -- the ECAL's
outer edge for an EM particle, the HCAL's for a hadron, the world radius for a
muon -- and any crossing beyond that point is dropped. That single rule is why
only muons reach the muon system.

**Neutral particles leave no position hits.** A tracking or muon chamber
measures ionization, which a neutral particle does not produce; a photon
crosses the silicon invisibly and is seen for the first time when it showers
in the calorimeter. Particles with no species at all (a bare kinematic table)
are exempt -- they are charged stubs by construction.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from detector2d.calorimeter import CaloRing
from detector2d.field import signed_radius
from detector2d.geometry import CircleLayer, Trajectory
from detector2d.propagate import SegmentedTrajectory, first_intersection_path, propagate

from . import response as response_module
from . import species as species_module
from .config import Layer, ParticleGunConfig, SimConfig
from .edm import DEPOSITS_COLUMNS, HITS_COLUMNS, PARTICLES_COLUMNS


def trajectory_for_row(row) -> Trajectory:
    """The single-arc trajectory a row describes, from its stored ``radius``.
    This is the constant-field path; see :func:`path_for_row` for the
    piecewise-field generalization."""
    radius = row["radius"]
    r = None if radius is None or (isinstance(radius, float) and math.isnan(radius)) else radius
    return Trajectory(x0=row["x0"], y0=row["y0"], phi0=row["phi0"], radius=r)


def path_for_row(
    row,
    field=None,
    world_radius: float | None = None,
    max_path_length: float | None = None,
) -> SegmentedTrajectory | Trajectory:
    """The particle's path: a plain :class:`~detector2d.geometry.Trajectory`
    in a constant field, or a multi-segment
    :class:`~detector2d.propagate.SegmentedTrajectory` when ``field`` is a
    :class:`~detector2d.field.FieldRegions`.

    Falls back to the single-arc form if the row carries no usable ``energy``
    -- rebuilding a piecewise path needs the momentum, not just the bend
    radius one particular region produced.
    """
    if field is None:
        return trajectory_for_row(row)
    energy = row.get("energy") if hasattr(row, "get") else None
    if energy is None or (isinstance(energy, float) and math.isnan(energy)):
        return trajectory_for_row(row)
    return propagate(
        x0=row["x0"],
        y0=row["y0"],
        phi0=row["phi0"],
        charge=row["charge"],
        pt=float(energy),
        field=field,
        world_radius=world_radius,
        max_path_length=max_path_length,
    )


def boundary_crossing_s(path, tracker_boundary: float | None) -> float | None:
    """Arc length at which ``path`` first crosses a circle of radius
    ``tracker_boundary`` centered at the origin, or ``None`` if there is no
    boundary set or the path never reaches it (e.g. a tight low-``pt`` curl
    that stays inside it). Reuses :mod:`detector2d.intersect` rather than
    re-deriving circle-crossing math: the boundary is just another
    :class:`~detector2d.geometry.CircleLayer`. Accepts either kind of path."""
    if tracker_boundary is None:
        return None
    boundary = CircleLayer(layer_id=-1, center=(0.0, 0.0), radius=tracker_boundary)
    hit = first_intersection_path(path, boundary)
    return hit.s if hit is not None else None


def system_outer_radius(layers, system: str) -> float | None:
    """Outer edge of a calorimeter, derived from its own rings rather than
    re-stated in config -- the radius past which a particle absorbed there
    cannot be seen."""
    rings = [layer for layer in layers if isinstance(layer, CaloRing) and layer.system == system]
    if not rings:
        return None
    return max(ring.radius + 0.5 * ring.thickness for ring in rings)


def stopping_radius(species, layers, response_config, world_radius: float | None) -> float | None:
    """The radius at which a particle of this species is absorbed: the EM
    calorimeter's outer edge for an EM particle, the hadronic one's for a
    hadron, and nothing short of the world for a muon (or for a species-less
    stub, which has no interaction model at all)."""
    if species is None or response_config is None:
        return world_radius
    if species.interaction == species_module.EM:
        return system_outer_radius(layers, response_config.em_system) or world_radius
    if species.interaction == species_module.HADRON:
        return system_outer_radius(layers, response_config.hadron_system) or world_radius
    return world_radius


def _sensitive_layers(layers) -> list[Layer]:
    """Layers that record a position hit -- everything except the calorimeter
    rings, which record energy instead (see :mod:`detectorsim2d.response`)."""
    return [layer for layer in layers if not isinstance(layer, CaloRing)]


def propagate_particles(
    particles_df: pd.DataFrame,
    layers: list[Layer],
    tracker_boundary: float | None = None,
    field=None,
    world_radius: float | None = None,
    max_path_length: float | None = None,
    response_config=None,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Propagate every particle through every layer, returning
    ``(hits, deposits)``.

    Hits: the earliest crossing of each sensitive (non-calorimeter) layer,
    for charged particles only, up to the particle's stopping radius.
    Deposits: the calorimeter response for the particle's species (see
    :mod:`detectorsim2d.response`), empty when no ``response_config`` is given.

    ``tracker_boundary`` still caps propagation as it always did: without it a
    curved trajectory's own circular arc loops back through the radius it
    started from, which is not physical -- the field that bends it exists only
    inside the detector, so a real particle leaving wouldn't curl back in.
    """
    sensitive = _sensitive_layers(layers)
    hit_rows: list[dict] = []
    deposit_rows: list[dict] = []
    hit_id = 0

    for _, particle in particles_df.iterrows():
        species = species_module.for_row(particle)
        path = path_for_row(particle, field, world_radius, max_path_length)

        cutoffs = [
            boundary_crossing_s(path, tracker_boundary),
            boundary_crossing_s(path, stopping_radius(species, layers, response_config, world_radius)),
        ]
        finite = [s for s in cutoffs if s is not None]
        stop_s = min(finite) if finite else None

        # a neutral particle does not ionize, so it leaves no position hits
        if species is None or species.is_charged:
            for layer in sensitive:
                hit = first_intersection_path(path, layer)
                if hit is None:
                    continue
                if stop_s is not None and hit.s > stop_s:
                    continue
                hit_rows.append(
                    dict(
                        event_id=particle["event_id"],
                        particle_id=particle["particle_id"],
                        system=layer.system,
                        layer_id=layer.layer_id,
                        hit_id=hit_id,
                        x=hit.x,
                        y=hit.y,
                        s_local=hit.local_coord,
                        path_length=hit.s,
                    )
                )
                hit_id += 1

        if response_config is not None:
            deposit_rows.extend(response_module.respond(particle, path, layers, response_config, rng))

    return (
        pd.DataFrame(hit_rows, columns=HITS_COLUMNS),
        pd.DataFrame(deposit_rows, columns=DEPOSITS_COLUMNS),
    )


def hits_for_particles(
    particles_df: pd.DataFrame,
    layers: list[Layer],
    tracker_boundary: float | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Just the hits from :func:`propagate_particles` -- the pre-calorimeter
    entry point, unchanged for callers that only track particles."""
    hits, _deposits = propagate_particles(particles_df, layers, tracker_boundary, **kwargs)
    return hits


def _sample_particle(rng: np.random.Generator, gun: ParticleGunConfig, field_bz: float, k: float) -> dict:
    x0 = gun.vertex_x + (rng.uniform(-gun.vertex_spread_x, gun.vertex_spread_x) if gun.vertex_spread_x else 0.0)
    y0 = gun.vertex_y + (rng.uniform(-gun.vertex_spread_y, gun.vertex_spread_y) if gun.vertex_spread_y else 0.0)
    phi0 = rng.uniform(gun.phi_min, gun.phi_max)
    pt = rng.uniform(gun.pt_min, gun.pt_max)

    if gun.species:
        name = str(rng.choice(gun.species, p=gun.species_probabilities))
        species = species_module.get(name)
        charge, pdg = species.charge, species.pdg
    else:
        # legacy, species-free gun: a bare charged stub, no calorimeter
        # response. NaN rather than None so the column survives a CSV round
        # trip unchanged (pandas reads an empty CSV field back as NaN).
        name, pdg = np.nan, np.nan
        charge = float(rng.choice(gun.charges))

    radius = signed_radius(pt, charge, field_bz, k)
    return dict(
        species=name, pdg=pdg, x0=x0, y0=y0, phi0=phi0, charge=charge, energy=pt, radius=radius
    )


def sample_particles(rng: np.random.Generator, config: SimConfig, event_id: int) -> list[dict]:
    return [
        dict(
            event_id=event_id,
            particle_id=particle_id,
            **_sample_particle(rng, config.gun, config.magnetic_field.bz, config.magnetic_field.k),
        )
        for particle_id in range(config.gun.n_particles)
    ]


def simulate_events(
    config: SimConfig, rng: np.random.Generator | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run config.n_events events. Returns (particles_df, hits_df, deposits_df)."""
    if rng is None:
        rng = np.random.default_rng(config.seed)
    particle_rows: list[dict] = []
    for event_id in range(config.n_events):
        particle_rows.extend(sample_particles(rng, config, event_id))
    particles_df = pd.DataFrame(particle_rows, columns=PARTICLES_COLUMNS)
    hits_df, deposits_df = propagate_particles(
        particles_df,
        config.layers,
        tracker_boundary=config.tracker_boundary,
        field=config.field_regions,
        world_radius=config.world_radius,
        max_path_length=config.max_path_length,
        response_config=config.response,
        rng=rng,
    )
    return particles_df, hits_df, deposits_df
