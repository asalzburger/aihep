"""End-to-end glue: `detectorsim2d` (jets/b-jets simulation) into
`detectorreco2d` (track/cluster reconstruction), plus a per-jet truth
summary the validation plots are built from.

This module owns no physics of its own -- everything b-jet-specific (extra
tracks, higher pt, the muon fraction) lives in `detectorsim2d`'s
`ParticleGunConfig` (see `configs/jets_bjets.yaml`); everything smearing
-specific lives in `detectorreco2d` (see `configs/reco.yaml`). What's here is
just: load the two configs, run both stages, and fold the reconstructed
tracks back into "how many tracks/muons did this jet get" -- the observables
`vis.py`'s plots actually show.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from detectorreco2d.config import RecoConfig
from detectorreco2d.config import load_config as load_reco_config
from detectorreco2d.reconstruct import reconstruct
from detectorsim2d.config import SimConfig
from detectorsim2d.config import load_config as load_sim_config
from detectorsim2d.simulate import simulate_events

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
SIM_CONFIG_PATH = CONFIGS / "jets_bjets.yaml"
RECO_CONFIG_PATH = CONFIGS / "reco.yaml"

MUON_SPECIES = ("mu-", "mu+")


def simulate_jets(
    config: SimConfig | None = None, rng: np.random.Generator | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate a run of jet events (light + b-jets mixed in) -- `(particles,
    hits, deposits)`, straight from `detectorsim2d.simulate.simulate_events`
    with `configs/jets_bjets.yaml` as the default config."""
    config = config if config is not None else load_sim_config(SIM_CONFIG_PATH)
    return simulate_events(config, rng=rng)


def reconstruct_run(
    particles: pd.DataFrame,
    deposits: pd.DataFrame,
    config: RecoConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reconstruct a simulated run -- `(tracks, clusters)`, with
    `configs/reco.yaml` as the default config."""
    config = config if config is not None else load_reco_config(RECO_CONFIG_PATH)
    return reconstruct(particles, deposits, config, rng=rng)


def run_pipeline(
    sim_config: SimConfig | None = None,
    reco_config: RecoConfig | None = None,
    seed: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate + reconstruct in one call: `(particles, hits, deposits,
    tracks, clusters)`. `seed`, if given, overrides both configs' own seeds
    -- convenient for a single reproducible end-to-end run."""
    sim_config = sim_config if sim_config is not None else load_sim_config(SIM_CONFIG_PATH)
    reco_config = reco_config if reco_config is not None else load_reco_config(RECO_CONFIG_PATH)
    if seed is not None:
        sim_config.seed = seed
        reco_config.seed = seed

    particles, hits, deposits = simulate_events(sim_config)
    tracks, clusters = reconstruct(particles, deposits, reco_config)
    return particles, hits, deposits, tracks, clusters


def summarize_jets(tracks: pd.DataFrame) -> pd.DataFrame:
    """One row per (`event_id`, `jet_id`) that actually belongs to a jet
    (`jet_id != -1`): its `is_b_jet` flag, its reconstructed track count
    (`n_tracks`), and how many of those tracks are muons (`n_muons`, by
    truth `species` -- see the module docstring of
    `detectorreco2d.reconstruct` on why that's fair game here even though a
    real reconstruction would have to earn it from muon-system hits
    instead).

    This is truth-level bookkeeping (`is_b_jet`/`species` are passthrough
    columns on `tracks`, not something derived from the smeared quantities),
    used only for the validation plots -- not a tagging algorithm.
    """
    if not len(tracks):
        return pd.DataFrame(columns=["event_id", "jet_id", "is_b_jet", "n_tracks", "n_muons"])

    jetted = tracks[tracks["jet_id"] != -1].copy()
    if not len(jetted):
        return pd.DataFrame(columns=["event_id", "jet_id", "is_b_jet", "n_tracks", "n_muons"])

    jetted["is_muon"] = jetted["species"].isin(MUON_SPECIES)
    summary = jetted.groupby(["event_id", "jet_id"], as_index=False).agg(
        is_b_jet=("is_b_jet", "first"),
        n_tracks=("particle_id", "count"),
        n_muons=("is_muon", "sum"),
    )
    return summary
