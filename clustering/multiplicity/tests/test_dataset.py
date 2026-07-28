import numpy as np
import pytest

from conftest import build_synthetic_run
from multiplicity.dataset import build_dataset, compute_matrix_shape
from multiplicity.io import read_run


def test_compute_matrix_shape_is_max_span_plus_one(tmp_path):
    build_synthetic_run(tmp_path, {1: 2, 2: 2, 3: 2})
    hits, clusters, _truth, contributions = read_run(tmp_path, "arrow")

    n_x, n_y = compute_matrix_shape(hits, clusters, contributions)

    assert n_x == 3 + 1  # largest span is the 3-particle cluster (x_span_pixels=3)
    assert n_y == 1 + 1  # every cluster here is 1 pixel tall


def test_build_dataset_labels_match_n_particles(tmp_path):
    build_synthetic_run(tmp_path, {1: 5, 2: 5, 3: 5})
    dataset = build_dataset([tmp_path], fmt="arrow")

    assert len(dataset.n_particles) == 15
    assert sorted(dataset.n_particles) == sorted([1] * 5 + [2] * 5 + [3] * 5)
    assert dataset.matrices.shape == (15, 4, 2)


def test_build_dataset_excludes_noise_only_clusters(tmp_path):
    build_synthetic_run(tmp_path, {1: 3})
    hits, clusters, truth, contributions = read_run(tmp_path, "arrow")

    # add a pure-noise cluster to the same event with no truth contribution
    import pandas as pd

    noise_hit = pd.DataFrame(
        [dict(event_id=0, ix=50, iy=50, x_center_um=1262.5, y_center_um=2525.0, charge=0.02, cluster_id=1)]
    )
    hits = pd.concat([hits, noise_hit], ignore_index=True)
    noise_cluster = pd.DataFrame(
        [
            dict(
                event_id=0,
                cluster_id=1,
                n_pixels=1,
                charge_sum=0.02,
                x_centroid_um=1262.5,
                y_centroid_um=2525.0,
                x_centroid_digital_um=1262.5,
                y_centroid_digital_um=2525.0,
                x_span_pixels=1,
                y_span_pixels=1,
            )
        ]
    )
    clusters = pd.concat([clusters, noise_cluster], ignore_index=True)

    from clustering_utils.io import write_tables

    write_tables(tmp_path, "arrow", {"hits": hits, "clusters": clusters, "truth": truth, "contributions": contributions})

    dataset = build_dataset([tmp_path], fmt="arrow")

    # still only the 3 real (particle-1) clusters -- the noise cluster is dropped
    assert len(dataset.n_particles) == 3
    assert (dataset.n_particles == 1).all()


def test_cluster_matrix_is_centered_and_conserves_charge(tmp_path):
    build_synthetic_run(tmp_path, {1: 1})
    dataset = build_dataset([tmp_path], fmt="arrow", matrix_shape=(5, 3))

    matrix = dataset.matrices[0]
    assert matrix.sum() == pytest.approx(1.0)
    # a single pixel (span 1) in a 5-wide/3-tall canvas centers at (2, 1)
    assert matrix[2, 1] == pytest.approx(1.0)


def test_build_dataset_respects_fixed_matrix_shape_across_dirs(tmp_path):
    dir_a = build_synthetic_run(tmp_path / "a", {1: 2})
    dir_b = build_synthetic_run(tmp_path / "b", {3: 2})

    # matrix_shape computed from dir_a alone would be too small for dir_b's
    # 3-particle clusters (span 3 > shape (2,2)) -- passing it explicitly
    # should surface that mismatch rather than silently truncating.
    with pytest.raises(ValueError):
        build_dataset([dir_a, dir_b], fmt="arrow", matrix_shape=(2, 2))

    dataset = build_dataset([dir_a, dir_b], fmt="arrow")
    assert dataset.matrix_shape == (4, 2)
    assert set(np.unique(dataset.source)) == {str(dir_a), str(dir_b)}
