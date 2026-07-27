"""Digitize continuous hit positions into per-layer integer cell indices.

`s_local` (a length, see `detector2d.intersect`) is divided by each layer's
`pitch` and floored. Layers with no `pitch` set are left un-digitized
(`cell_index = NaN`) -- their hits pass through but are never clustered.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .edm import HITS_COLUMNS


def digitize_hits(hits_df: pd.DataFrame, layers) -> pd.DataFrame:
    pitch_by_layer = {layer.layer_id: layer.pitch for layer in layers}
    out = hits_df.copy()
    pitches = out["layer_id"].map(pitch_by_layer).astype(float)
    out["cell_index"] = np.floor(out["s_local"] / pitches)
    return out[HITS_COLUMNS]
