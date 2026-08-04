import numpy as np
import pandas as pd
import pytest
from detectorreco2d.edm import CLUSTERS_COLUMNS, TRACKS_COLUMNS


def build_synthetic_reco_run(n_light: int = 60, n_b_jet: int = 60, seed: int = 0):
    """A tiny, trivially-separable synthetic reconstructed run: `n_light`
    light jets and `n_b_jet` b-jets, one per event, each with a clean
    flavor-discriminating signal (wider `d0`, more tracks, more often a
    muon, higher cluster energy for a b-jet -- see the module docstring of
    `flavor_tagging.dataset`) -- fast and deterministic, for exercising the
    b-tagger's training/evaluation machinery without running the full
    simulate+reconstruct pipeline."""
    rng = np.random.default_rng(seed)
    track_rows, cluster_rows = [], []
    particle_id = 0
    event_id = 0

    for is_b_jet, n_jets in ((False, n_light), (True, n_b_jet)):
        for _ in range(n_jets):
            jet_id = 0
            n_tracks = int(rng.integers(9, 13)) if is_b_jet else int(rng.integers(5, 9))
            d0_sigma = 4.0 if is_b_jet else 0.3
            has_muon = rng.random() < (0.5 if is_b_jet else 0.02)
            n_clusters = int(rng.integers(3, 6))
            energy_scale = 30.0 if is_b_jet else 10.0

            for i in range(n_tracks):
                species = "mu-" if (has_muon and i == 0) else "pi+"
                pt = float(rng.uniform(50.0, 300.0))
                track_rows.append(
                    dict(
                        event_id=event_id,
                        particle_id=particle_id,
                        jet_id=jet_id,
                        is_b_jet=is_b_jet,
                        species=species,
                        charge=1.0,
                        d0_true=0.0,
                        d0=float(rng.normal(0.0, d0_sigma)),
                        phi0_true=0.0,
                        phi0=0.0,
                        pt_true=pt,
                        pt=pt,
                    )
                )
                particle_id += 1

            for _ in range(n_clusters):
                energy = float(rng.uniform(0.5, 1.5) * energy_scale)
                cluster_rows.append(
                    dict(
                        event_id=event_id,
                        particle_id=particle_id,
                        jet_id=jet_id,
                        is_b_jet=is_b_jet,
                        species="photon",
                        energy_true=energy,
                        energy=energy,
                    )
                )
                particle_id += 1

            event_id += 1

    tracks = pd.DataFrame(track_rows, columns=TRACKS_COLUMNS)
    clusters = pd.DataFrame(cluster_rows, columns=CLUSTERS_COLUMNS)
    return tracks, clusters


@pytest.fixture
def synthetic_reco_run():
    return build_synthetic_reco_run(seed=0)


@pytest.fixture
def independent_synthetic_reco_run():
    """A second synthetic run from a different seed -- stands in for the
    genuinely independent test dataset `evaluate_model` should be checked
    against, never the training set itself."""
    return build_synthetic_reco_run(seed=99)
