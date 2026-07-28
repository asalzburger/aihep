"""The pluggable splitter interface.

A `Splitter` decides, per hit pixel, which sub-group of its *original*
cluster it belongs to. Turning that decision into a fully self-consistent
(hits, clusters) pair -- renumbering cluster_id and recomputing cluster
aggregates -- is the same for every splitter and lives in `pipeline.py`.
Swapping in a new splitting strategy means writing one `Splitter` subclass
and registering it in `registry.SPLITTERS`; nothing else changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class Splitter(ABC):
    name: str

    @abstractmethod
    def split_key(self, hits: pd.DataFrame, clusters: pd.DataFrame, contributions: pd.DataFrame) -> pd.Series:
        """One label per row of `hits` (same length and row order).

        Pixels that share both the same original `cluster_id` *and* the
        same label stay together in the output; pixels in the same
        original cluster with different labels are split apart. Labels are
        only ever compared within a single (event_id, cluster_id) group,
        so any hashable value works and collisions across different
        original clusters (or events) don't matter -- pixels that
        shouldn't be split apart from the rest of their cluster can all
        share one constant label (e.g. 0).
        """
