"""Event generation: sample particles (or take a given ground truth) and
intersect them with a detector layout to produce hits.

Split in two on purpose:

- :func:`hits_for_particles` turns *any* particles table into hits by
  propagating each row through the layers. It doesn't care where the
  particles came from -- a random gun, or (as in the Denby recreation)
  parameters fit straight out of a reference picture.
- :func:`simulate_events` is the random-gun path: sample particles from a
  :class:`~tracksim2d.config.ParticleGunConfig`, then call
  :func:`hits_for_particles`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from detector2d.field import signed_radius
from detector2d.geometry import CircleLayer, Trajectory
from detector2d.intersect import first_intersection

from .config import Layer, ParticleGunConfig, SimConfig
from .edm import HITS_COLUMNS, PARTICLES_COLUMNS


def trajectory_for_row(row) -> Trajectory:
    radius = row["radius"]
    r = None if radius is None or (isinstance(radius, float) and math.isnan(radius)) else radius
    return Trajectory(x0=row["x0"], y0=row["y0"], phi0=row["phi0"], radius=r)


def boundary_crossing_s(trajectory: Trajectory, tracker_boundary: float | None) -> float | None:
    """Arc length at which ``trajectory`` first crosses a circle of radius
    ``tracker_boundary`` centered at the origin -- the tracker's outer
    boundary -- or ``None`` if there is no boundary set or the trajectory
    never reaches it (e.g. a tight low-``pt`` curl that stays inside it).
    Reuses :mod:`detector2d.intersect` rather than re-deriving circle-crossing
    math: the boundary is just another :class:`~detector2d.geometry.CircleLayer`."""
    if tracker_boundary is None:
        return None
    boundary = CircleLayer(layer_id=-1, center=(0.0, 0.0), radius=tracker_boundary)
    hit = first_intersection(trajectory, boundary)
    return hit.s if hit is not None else None


def hits_for_particles(
    particles_df: pd.DataFrame, layers: list[Layer], tracker_boundary: float | None = None
) -> pd.DataFrame:
    """Propagate every particle through every layer, keeping the earliest
    (smallest arc length) crossing of each layer as that particle's hit.

    If ``tracker_boundary`` is given, propagation stops once the particle
    first exits that radius (see :func:`boundary_crossing_s`): any crossing
    beyond it is dropped. Without this cutoff, a curved trajectory's own
    circular arc loops back through the same radius it started from (see
    :mod:`detector2d.barrel`'s module docstring), which is not physical --
    the constant field that bends it exists only inside the tracker volume,
    so a real particle leaving it wouldn't curl back in.
    """
    rows: list[dict] = []
    hit_id = 0
    for _, particle in particles_df.iterrows():
        trajectory = trajectory_for_row(particle)
        boundary_s = boundary_crossing_s(trajectory, tracker_boundary)
        for layer in layers:
            hit = first_intersection(trajectory, layer)
            if hit is None:
                continue
            if boundary_s is not None and hit.s > boundary_s:
                continue
            rows.append(
                dict(
                    event_id=particle["event_id"],
                    particle_id=particle["particle_id"],
                    layer_id=layer.layer_id,
                    hit_id=hit_id,
                    x=hit.x,
                    y=hit.y,
                    s_local=hit.local_coord,
                    path_length=hit.s,
                )
            )
            hit_id += 1
    return pd.DataFrame(rows, columns=HITS_COLUMNS)


def _sample_particle(rng: np.random.Generator, gun: ParticleGunConfig, field_bz: float, k: float) -> dict:
    x0 = gun.vertex_x + (rng.uniform(-gun.vertex_spread_x, gun.vertex_spread_x) if gun.vertex_spread_x else 0.0)
    y0 = gun.vertex_y + (rng.uniform(-gun.vertex_spread_y, gun.vertex_spread_y) if gun.vertex_spread_y else 0.0)
    phi0 = rng.uniform(gun.phi_min, gun.phi_max)
    charge = float(rng.choice(gun.charges))
    pt = rng.uniform(gun.pt_min, gun.pt_max)
    radius = signed_radius(pt, charge, field_bz, k)
    return dict(x0=x0, y0=y0, phi0=phi0, charge=charge, radius=radius)


def sample_particles(rng: np.random.Generator, config: SimConfig, event_id: int) -> list[dict]:
    return [
        dict(
            event_id=event_id,
            particle_id=particle_id,
            **_sample_particle(rng, config.gun, config.magnetic_field.bz, config.magnetic_field.k),
        )
        for particle_id in range(config.gun.n_particles)
    ]


def simulate_events(config: SimConfig, rng: np.random.Generator | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run config.n_events events. Returns (particles_df, hits_df)."""
    if rng is None:
        rng = np.random.default_rng(config.seed)
    particle_rows: list[dict] = []
    for event_id in range(config.n_events):
        particle_rows.extend(sample_particles(rng, config, event_id))
    particles_df = pd.DataFrame(particle_rows, columns=PARTICLES_COLUMNS)
    hits_df = hits_for_particles(particles_df, config.layers, tracker_boundary=config.tracker_boundary)
    return particles_df, hits_df
