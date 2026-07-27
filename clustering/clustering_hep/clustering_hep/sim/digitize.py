"""Turn a per-event raw charge grid into a hits table: optional Gaussian
diffusion + electronic noise, then threshold to decide which pixels register
as hits. All three effects default to no-ops (see DigitizationConfig)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from ..edm import HITS_COLUMNS
from .config import DetectorConfig, DigitizationConfig, SimConfig


def digitize_grid(
    grid: np.ndarray,
    detector: DetectorConfig,
    digitization: DigitizationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    out = grid
    if digitization.diffusion_sigma_um > 0:
        sigma = (
            digitization.diffusion_sigma_um / detector.pitch_x_um,
            digitization.diffusion_sigma_um / detector.pitch_y_um,
        )
        out = gaussian_filter(out, sigma=sigma)
    if digitization.noise_sigma > 0:
        out = out + rng.normal(0.0, digitization.noise_sigma, size=out.shape)
    return out


def grid_to_hit_rows(event_id: int, grid: np.ndarray, detector: DetectorConfig, threshold: float) -> list[dict]:
    ix_arr, iy_arr = np.nonzero(grid > threshold)
    return [
        dict(
            event_id=event_id,
            ix=int(ix),
            iy=int(iy),
            x_center_um=(ix + 0.5) * detector.pitch_x_um,
            y_center_um=(iy + 0.5) * detector.pitch_y_um,
            charge=float(grid[ix, iy]),
        )
        for ix, iy in zip(ix_arr, iy_arr)
    ]


def digitize_events(
    grids: dict[int, np.ndarray], config: SimConfig, rng: np.random.Generator | None = None
) -> pd.DataFrame:
    if rng is None:
        rng = np.random.default_rng(config.seed)
    rows: list[dict] = []
    for event_id, grid in grids.items():
        digitized = digitize_grid(grid, config.detector, config.digitization, rng)
        rows.extend(grid_to_hit_rows(event_id, digitized, config.detector, config.digitization.threshold))
    return pd.DataFrame(rows, columns=HITS_COLUMNS)
