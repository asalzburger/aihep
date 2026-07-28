"""An oracle splitter: uses ground truth (the `contributions` table) to
decide the split, rather than any reconstructed quantity. Not a real
reconstruction algorithm -- it's the reference/baseline other, genuinely
data-driven splitters should be validated against.
"""

from __future__ import annotations

import pandas as pd

from .base import Splitter

NO_CONTRIBUTION_KEY = -1
"""split_key given to a hit pixel with no entry in `contributions` at all
(e.g. a purely noise-driven hit, or an occasional noise pixel that happens
to sit next to a real cluster and gets glued on by connectivity) -- no
truth particle claims it, so it isn't merged into any particle's split."""


def dominant_particle_per_pixel(contributions: pd.DataFrame) -> pd.DataFrame:
    """One row per (event_id, ix, iy): the particle_id that deposited the
    most charge into that pixel."""
    idx = contributions.groupby(["event_id", "ix", "iy"])["charge"].idxmax()
    return contributions.loc[idx, ["event_id", "ix", "iy", "particle_id"]].reset_index(drop=True)


class TruthSplitter(Splitter):
    """Assigns each hit pixel to whichever truth particle deposited the
    most charge into it, then splits a cluster wherever that assignment
    differs across its pixels."""

    name = "truth"

    def split_key(self, hits: pd.DataFrame, clusters: pd.DataFrame, contributions: pd.DataFrame) -> pd.Series:
        dominant = dominant_particle_per_pixel(contributions)
        merged = hits[["event_id", "ix", "iy"]].merge(dominant, on=["event_id", "ix", "iy"], how="left")
        return merged["particle_id"].fillna(NO_CONTRIBUTION_KEY).astype(int)
