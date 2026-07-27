"""Tests for the Denby recreation: detector harmonization, the SVG-fitted
event, and the objective "do they line up" check standing in for the eyeball
overlay comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from denby_svg import derive_vertex, parse_event_tracks, parse_layer_lines, sample_cubic_path
from fit_event import EVENT_SVG, fit_event
from harmonize_detector import harmonized_layers
from render_event import RESOURCES

from tracksim2d.edm import PARTICLES_COLUMNS
from tracksim2d.io import read_table, write_table
from tracksim2d.simulate import hits_for_particles

MAX_PIXEL_RESIDUAL = 5.0  # px, on a ~1000px-wide figure -- see README.md


def test_original_detector_has_13_layers_same_length_but_uneven_spacing():
    original = parse_layer_lines(Path(EVENT_SVG).read_text())
    assert len(original) == 13
    lengths = {round(x2 - x1, 1) for x1, _, x2, _ in original}
    assert lengths == {709.5}  # already equal in the reference figure
    ys = [y for _, y, _, _ in original]
    spacings = {round(b - a, 1) for a, b in zip(ys[:-1], ys[1:])}
    assert len(spacings) > 1  # ...but not equidistant, which is what we harmonize


def test_harmonized_detector_is_13_equidistant_equal_length_layers():
    layers = harmonized_layers()
    assert len(layers) == 13

    lengths = {round(layer.length, 3) for layer in layers}
    assert len(lengths) == 1

    ys = [layer.p1[1] for layer in layers]
    spacings = {round(b - a, 3) for a, b in zip(ys[:-1], ys[1:])}
    assert len(spacings) == 1

    # every layer horizontal, at the same x-extent
    xs = {(layer.p1[0], layer.p2[0]) for layer in layers}
    assert len(xs) == 1


def test_derived_vertex_matches_the_marker_dot_and_the_circle_fit_cross_check():
    # derive_vertex() raises if the two independent methods disagree by more
    # than a few pixels -- calling it at all is the real assertion.
    x, y = derive_vertex(Path(EVENT_SVG).read_text())
    assert (x, y) == pytest.approx((424.25, 894.54), abs=0.1)


def test_fitted_event_has_one_row_per_track_sharing_the_vertex():
    particles = fit_event()
    assert list(particles.columns) == PARTICLES_COLUMNS
    assert len(particles) == 4
    assert particles["x0"].nunique() == 1
    assert particles["y0"].nunique() == 1
    assert set(particles["charge"]) <= {-1.0, 1.0}


@pytest.mark.parametrize("fmt", ["csv", "arrow"])
def test_denby_event_round_trips_through_tracksim2d_io(tmp_path, fmt):
    particles = fit_event()
    path = tmp_path / f"particles.{fmt}"
    write_table(particles, path, fmt)
    reloaded = read_table(path, fmt)
    pd.testing.assert_frame_equal(particles, reloaded, check_dtype=False)


def test_simulated_hits_match_the_reference_tracks_within_a_few_pixels():
    """The objective version of 'overlaid, it should match up pretty nicely':
    for each track, compare the simulated hit's x at a given y against the
    original reference curve's x at that same y."""
    layers = harmonized_layers()
    particles = fit_event()
    hits = hits_for_particles(particles, layers)

    svg_text = Path(EVENT_SVG).read_text()
    tracks = parse_event_tracks(svg_text)
    assert len(tracks) == len(particles)

    for particle_id, d in enumerate(tracks):
        points = sample_cubic_path(d, n_per_segment=200)
        order = points[:, 1].argsort()
        ys, xs = points[order, 1], points[order, 0]

        particle_hits = hits[hits["particle_id"] == particle_id]
        in_range = particle_hits[(particle_hits["y"] >= ys.min()) & (particle_hits["y"] <= ys.max())]
        assert len(in_range) > 0, f"particle {particle_id} has no hits within its reference track's y-range"

        x_reference = np.interp(in_range["y"], ys, xs)
        residual = np.abs(x_reference - in_range["x"].to_numpy())
        assert residual.max() < MAX_PIXEL_RESIDUAL


def test_generated_resources_exist_and_look_like_svg():
    for name in (
        "denby_layers.csv",
        "denby_event.csv",
        "denby_detector_harmonized.svg",
        "denby_event_simulated.svg",
        "denby_overlay.svg",
    ):
        path = RESOURCES / name
        assert path.exists(), f"missing {path} -- run harmonize_detector.py / fit_event.py / render_event.py"
        if name.endswith(".svg"):
            assert path.read_text().startswith("<?xml")


def test_hopfield_tracking_reconstructs_most_of_the_real_event():
    """Integration check that the separate hopfield_tracking package
    actually converges to a sensible reconstruction of this page's own
    event -- not just that the driver script ran once and left files
    behind. See hopfield_tracking/README.md for the full, honest story
    (2 of 4 tracks exact, the other 2 mostly right, confusion right at the
    shared vertex)."""
    from hopfield_tracking.cli import run as hopfield_run

    from tracksim2d.simulate import hits_for_particles

    layers = harmonized_layers()
    particles = fit_event()
    hits = hits_for_particles(particles, layers)[["particle_id", "layer_id", "x", "y"]]

    _, history, _, score = hopfield_run(hits, seed=3)
    assert len(history) > 1
    assert score["n_true_tracks"] == 4
    assert score["n_exact_matches"] >= 2


def test_hopfield_driver_resources_exist():
    for name in ("denby_hopfield_hits.csv", "denby_hopfield_fig8.png", "denby_random_events.csv", "denby_sweep_results.csv"):
        path = RESOURCES / name
        assert path.exists(), (
            f"missing {path} -- run run_hopfield_on_denby_event.py / generate_random_events.py / run_hopfield_sweep.py"
        )
