"""Cluster splitting: given a `sensor` run whose clusters may merge more
than one truth particle's hits, decide how to split a cluster back apart
into per-particle sub-clusters.

`base.Splitter` is the pluggable interface; `pipeline.apply_splitter` turns
any Splitter's decision into a self-consistent (hits, clusters) pair;
`registry.SPLITTERS` is where new splitters get registered for `--splitter`
on the CLI.
"""

from __future__ import annotations

from .base import Splitter
from .pipeline import apply_splitter
from .registry import SPLITTERS
from .truth_splitter import TruthSplitter, dominant_particle_per_pixel

__all__ = [
    "Splitter",
    "apply_splitter",
    "SPLITTERS",
    "TruthSplitter",
    "dominant_particle_per_pixel",
]
