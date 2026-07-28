from pathlib import Path

import pandas as pd
import pytest
from clustering_utils.io import write_tables


def _cluster_rows(event_id: int, n_particles: int) -> dict:
    ix = list(range(n_particles))
    x_centers = [(i + 0.5) * 25.0 for i in ix]
    return dict(
        event_id=event_id,
        cluster_id=0,
        n_pixels=n_particles,
        charge_sum=float(n_particles),
        x_centroid_um=sum(x_centers) / n_particles,
        y_centroid_um=25.0,
        x_centroid_digital_um=sum(x_centers) / n_particles,
        y_centroid_digital_um=25.0,
        x_span_pixels=n_particles,
        y_span_pixels=1,
    )


def build_synthetic_run(output_dir: Path, counts_per_class: dict[int, int], fmt: str = "arrow") -> Path:
    """A tiny synthetic sensor-shaped run: one cluster per event, made of
    `n_particles` adjacent pixels (ix=0..n_particles-1, iy=0), one particle
    per pixel -- x_span_pixels == n_particles by construction, so the
    dataset's label (number of distinct contributing particles) exactly
    matches n_particles, same as the real p1/p2/p3/p123 data."""
    hits_rows, truth_rows, contribution_rows, cluster_rows = [], [], [], []
    event_id = 0
    for n_particles, count in counts_per_class.items():
        for _ in range(count):
            for particle_id in range(n_particles):
                hits_rows.append(
                    dict(
                        event_id=event_id,
                        ix=particle_id,
                        iy=0,
                        x_center_um=(particle_id + 0.5) * 25.0,
                        y_center_um=25.0,
                        charge=1.0,
                        cluster_id=0,
                    )
                )
                contribution_rows.append(
                    dict(event_id=event_id, particle_id=particle_id, ix=particle_id, iy=0, charge=1.0)
                )
                truth_rows.append(
                    dict(
                        event_id=event_id,
                        particle_id=particle_id,
                        x0_um=0.0,
                        y0_um=0.0,
                        dxdz=0.0,
                        dydz=0.0,
                        charge_deposited=1.0,
                        path_length_um=150.0,
                    )
                )
            cluster_rows.append(_cluster_rows(event_id, n_particles))
            event_id += 1

    tables = {
        "hits": pd.DataFrame(hits_rows),
        "clusters": pd.DataFrame(cluster_rows),
        "truth": pd.DataFrame(truth_rows),
        "contributions": pd.DataFrame(contribution_rows),
    }
    write_tables(output_dir, fmt, tables)
    return output_dir


@pytest.fixture
def synthetic_run(tmp_path):
    return build_synthetic_run(tmp_path, {1: 20, 2: 20, 3: 20})
