"""Configuration dataclasses for the pixel-detector cluster simulation.

All fields default to the values specified in the project brief, so
``SimConfig()`` with no YAML file reproduces the documented defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class DetectorConfig:
    thickness_um: float = 150.0
    pitch_x_um: float = 25.0
    pitch_y_um: float = 50.0
    n_pixels_x: int = 200
    n_pixels_y: int = 200
    lorentz_slope: float = 0.0  # dx/dz-equivalent constant drift; 0 = no drift


@dataclass
class ParticleConfig:
    angle_spread: float = 0.3  # uniform: max |dxdz|,|dydz|; gauss: sigma
    angle_distribution: Literal["uniform", "gauss"] = "uniform"
    nominal_dxdz: float = 0.0
    nominal_dydz: float = 0.0
    charge_per_um: float = 1.0 / 150.0  # k, so a perpendicular track deposits charge = 1.0


@dataclass
class MultiParticleConfig:
    n_particles: int = 1
    opening_angle_x: float = 0.05
    opening_angle_y: float = 0.05
    opening_distribution: Literal["uniform", "gauss", "exponential"] = "uniform"


@dataclass
class DigitizationConfig:
    diffusion_sigma_um: float = 0.0  # 0 disables Gaussian blur of the charge grid
    noise_sigma: float = 0.0  # 0 disables per-pixel additive noise
    threshold: float = 0.0  # pixels with charge <= threshold are not hits


@dataclass
class SimConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    particle: ParticleConfig = field(default_factory=ParticleConfig)
    multi: MultiParticleConfig = field(default_factory=MultiParticleConfig)
    digitization: DigitizationConfig = field(default_factory=DigitizationConfig)
    n_events: int = 1
    seed: int | None = None
    cluster_connectivity: Literal[4, 8] = 8


def _merge_dataclass(cls: type, data: dict[str, Any] | None):
    """Build a flat dataclass (no nested dataclass fields) from a dict,
    keeping the dataclass default for any key not present in ``data``."""
    data = data or {}
    known = {f.name for f in fields(cls)}
    kwargs = {k: v for k, v in data.items() if k in known}
    return cls(**kwargs)


_SECTION_TYPES = {
    "detector": DetectorConfig,
    "particle": ParticleConfig,
    "multi": MultiParticleConfig,
    "digitization": DigitizationConfig,
}


def load_config(path: str | Path | None) -> SimConfig:
    """Load a SimConfig from a YAML file, falling back to defaults for
    anything not present. ``path=None`` returns pure defaults."""
    if path is None:
        return SimConfig()

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    kwargs: dict[str, Any] = {}
    for section, cls in _SECTION_TYPES.items():
        if section in raw:
            kwargs[section] = _merge_dataclass(cls, raw[section])
    for key in ("n_events", "seed", "cluster_connectivity"):
        if key in raw:
            kwargs[key] = raw[key]

    return SimConfig(**kwargs)
