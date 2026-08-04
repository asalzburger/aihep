"""Configuration dataclasses for the 2D track/cluster reconstruction.

All fields default to zero resolution (``a=0, b=0``, i.e. no smearing at
all -- reconstructed values equal truth exactly), so ``RecoConfig()`` with no
YAML file still works and is the natural "unit test" baseline. Follows the
same load/merge pattern as `detectorsim2d/config.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Resolution:
    """One quantity's Gaussian smearing width: ``sigma = a + b / x``, where
    ``x`` is that track's `pt` (for `d0`/`phi0`/`pt` itself) or that
    cluster's `energy` -- the higher ``x``, the smaller ``sigma``, shrinking
    toward the asymptotic floor ``a``. See
    :func:`detectorreco2d.reconstruct.resolution`.
    """

    a: float = 0.0
    b: float = 0.0


@dataclass
class TrackResolution:
    d0: Resolution = field(default_factory=Resolution)
    phi0: Resolution = field(default_factory=Resolution)
    pt: Resolution = field(default_factory=Resolution)


@dataclass
class ClusterResolution:
    energy: Resolution = field(default_factory=Resolution)


@dataclass
class RecoConfig:
    track: TrackResolution = field(default_factory=TrackResolution)
    cluster: ClusterResolution = field(default_factory=ClusterResolution)
    #: RNG seed for the smearing draws, independent of whatever seed the
    #: upstream `detectorsim2d` simulation used.
    seed: int | None = None


def _parse_resolution(raw: dict[str, Any] | None) -> Resolution:
    raw = raw or {}
    return Resolution(a=float(raw.get("a", 0.0)), b=float(raw.get("b", 0.0)))


def load_config(path: str | Path | None) -> RecoConfig:
    """Load a RecoConfig from a YAML file, falling back to defaults (no
    smearing) for anything not present. ``path=None`` returns pure
    defaults::

        track_resolution:
          d0:   {a: 0.05, b: 2.0}
          phi0: {a: 0.001, b: 0.02}
          pt:   {a: 0.1, b: 1.0}
        cluster_resolution:
          energy: {a: 0.1, b: 3.0}
        seed: 123
    """
    if path is None:
        return RecoConfig()

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    track_raw = raw.get("track_resolution", {}) or {}
    cluster_raw = raw.get("cluster_resolution", {}) or {}

    kwargs: dict[str, Any] = {
        "track": TrackResolution(
            d0=_parse_resolution(track_raw.get("d0")),
            phi0=_parse_resolution(track_raw.get("phi0")),
            pt=_parse_resolution(track_raw.get("pt")),
        ),
        "cluster": ClusterResolution(energy=_parse_resolution(cluster_raw.get("energy"))),
    }
    if "seed" in raw:
        kwargs["seed"] = raw["seed"]

    return RecoConfig(**kwargs)
