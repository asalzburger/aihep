"""Simulation: detector/particle configuration, track geometry, charge
digitization, and cluster finding."""

from .clustering import cluster_hits
from .config import (
    DetectorConfig,
    DigitizationConfig,
    MultiParticleConfig,
    ParticleConfig,
    SimConfig,
    load_config,
)
from .digitize import digitize_events, digitize_grid, grid_to_hit_rows
from .geometry import (
    charge_endpoints,
    deposited_charge,
    path_length_through_slab,
    segment_pixel_fractions,
    true_center_position,
)
from .simulate import simulate_event, simulate_events

__all__ = [
    "cluster_hits",
    "DetectorConfig",
    "DigitizationConfig",
    "MultiParticleConfig",
    "ParticleConfig",
    "SimConfig",
    "load_config",
    "digitize_events",
    "digitize_grid",
    "grid_to_hit_rows",
    "charge_endpoints",
    "deposited_charge",
    "path_length_through_slab",
    "segment_pixel_fractions",
    "true_center_position",
    "simulate_event",
    "simulate_events",
]
