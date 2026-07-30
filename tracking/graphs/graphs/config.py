"""YAML-driven configuration for graph building -- the same load/merge
pattern as `tracksim2d.config.load_config`: a single `GraphConfig` dataclass
with a sensible default, loaded from a `prescription:` section in a config
file. See `graphs.prescription` for what `kind`s are available and
`configs/` for one working example per kind.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .prescription import FullyConnected, Prescription, parse_prescription


@dataclass
class GraphConfig:
    prescription: Prescription = field(default_factory=FullyConnected)


def load_config(path: str | Path | None) -> GraphConfig:
    """Load a `GraphConfig` from a YAML file. ``path=None``, or a file with
    no `prescription:` key, falls back to `FullyConnected()`."""
    if path is None:
        return GraphConfig()

    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    if "prescription" not in raw:
        return GraphConfig()
    return GraphConfig(prescription=parse_prescription(raw["prescription"]))
