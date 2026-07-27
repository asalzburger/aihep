"""Event generation: sample particle vertices/directions and deposit their
charge onto a per-event pixel grid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..edm import TRUTH_COLUMNS
from .config import MultiParticleConfig, ParticleConfig, SimConfig
from .geometry import charge_endpoints, deposited_charge, path_length_through_slab, segment_pixel_fractions


def _sample_direction(rng: np.random.Generator, particle: ParticleConfig) -> tuple[float, float]:
    if particle.angle_distribution == "uniform":
        dxdz = particle.nominal_dxdz + rng.uniform(-particle.angle_spread, particle.angle_spread)
        dydz = particle.nominal_dydz + rng.uniform(-particle.angle_spread, particle.angle_spread)
    elif particle.angle_distribution == "gauss":
        dxdz = rng.normal(particle.nominal_dxdz, particle.angle_spread)
        dydz = rng.normal(particle.nominal_dydz, particle.angle_spread)
    else:
        raise ValueError(f"Unknown angle_distribution: {particle.angle_distribution!r}")
    return dxdz, dydz


def _sample_opening_offset(rng: np.random.Generator, multi: MultiParticleConfig) -> tuple[float, float]:
    dist = multi.opening_distribution
    if dist == "uniform":
        return (
            rng.uniform(-multi.opening_angle_x, multi.opening_angle_x),
            rng.uniform(-multi.opening_angle_y, multi.opening_angle_y),
        )
    if dist == "gauss":
        return rng.normal(0.0, multi.opening_angle_x), rng.normal(0.0, multi.opening_angle_y)
    if dist == "exponential":
        sign_x, sign_y = rng.choice((-1.0, 1.0)), rng.choice((-1.0, 1.0))
        return (
            sign_x * rng.exponential(multi.opening_angle_x),
            sign_y * rng.exponential(multi.opening_angle_y),
        )
    raise ValueError(f"Unknown opening_distribution: {dist!r}")


def _sample_vertex(rng: np.random.Generator, config: SimConfig) -> tuple[float, float]:
    detector = config.detector
    x0 = rng.uniform(0.0, detector.n_pixels_x * detector.pitch_x_um)
    y0 = rng.uniform(0.0, detector.n_pixels_y * detector.pitch_y_um)
    return x0, y0


def simulate_event(rng: np.random.Generator, config: SimConfig, event_id: int) -> tuple[np.ndarray, list[dict]]:
    """Simulate one event. Returns (charge_grid[n_pixels_x, n_pixels_y], truth_rows)."""
    detector, particle, multi = config.detector, config.particle, config.multi
    grid = np.zeros((detector.n_pixels_x, detector.n_pixels_y), dtype=float)

    x0, y0 = _sample_vertex(rng, config)
    dxdz_nom, dydz_nom = _sample_direction(rng, particle)

    truth_rows = []
    for particle_id in range(multi.n_particles):
        off_x, off_y = _sample_opening_offset(rng, multi) if multi.n_particles > 1 else (0.0, 0.0)
        dxdz, dydz = dxdz_nom + off_x, dydz_nom + off_y

        p0, p1 = charge_endpoints(x0, y0, dxdz, dydz, detector.thickness_um, detector.lorentz_slope)
        q_total = deposited_charge(dxdz, dydz, detector.thickness_um, particle.charge_per_um)
        fractions = segment_pixel_fractions(
            p0, p1, detector.pitch_x_um, detector.pitch_y_um, detector.n_pixels_x, detector.n_pixels_y
        )
        for (ix, iy), frac in fractions.items():
            grid[ix, iy] += q_total * frac

        truth_rows.append(
            dict(
                event_id=event_id,
                particle_id=particle_id,
                x0_um=x0,
                y0_um=y0,
                dxdz=dxdz,
                dydz=dydz,
                charge_deposited=q_total,
                path_length_um=path_length_through_slab(dxdz, dydz, detector.thickness_um),
            )
        )

    return grid, truth_rows


def simulate_events(
    config: SimConfig, rng: np.random.Generator | None = None
) -> tuple[dict[int, np.ndarray], pd.DataFrame]:
    """Run config.n_events events.

    Returns ({event_id: charge_grid}, truth_df).
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)
    grids: dict[int, np.ndarray] = {}
    truth_rows: list[dict] = []
    for event_id in range(config.n_events):
        grid, rows = simulate_event(rng, config, event_id)
        grids[event_id] = grid
        truth_rows.extend(rows)
    truth_df = pd.DataFrame(truth_rows, columns=TRUTH_COLUMNS)
    return grids, truth_df
