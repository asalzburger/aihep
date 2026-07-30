import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from viz_style import PRESENT, PRINT

from sensor.edm import CLUSTERS_COLUMNS, CLUSTERED_HITS_COLUMNS, TRUTH_COLUMNS
from sensor.sim.config import DetectorConfig
from sensor.vis import (
    CHARGE_CMAP,
    ENTRY_MARKER,
    EXIT_MARKER,
    _clip_window,
    _footprint_center,
    _zoom_pixel_bounds,
    plot_event,
    plot_event_3d,
)


def _one_pixel_event():
    detector = DetectorConfig()
    hits = pd.DataFrame(
        [dict(event_id=0, ix=10, iy=10, x_center_um=250.0, y_center_um=500.0, charge=1.0, cluster_id=0)],
        columns=CLUSTERED_HITS_COLUMNS,
    )
    clusters = pd.DataFrame(
        [
            dict(
                event_id=0, cluster_id=0, n_pixels=1, charge_sum=1.0,
                x_centroid_um=250.0, y_centroid_um=500.0,
                x_centroid_digital_um=250.0, y_centroid_digital_um=500.0,
                x_span_pixels=1, y_span_pixels=1,
            )
        ],
        columns=CLUSTERS_COLUMNS,
    )
    truth = pd.DataFrame(
        [dict(event_id=0, particle_id=0, x0_um=250.0, y0_um=500.0, dxdz=0.0, dydz=0.0, charge_deposited=1.0, path_length_um=150.0)],
        columns=TRUTH_COLUMNS,
    )
    return hits, clusters, truth, detector


def test_plot_event_print_keeps_title_axes_colorbar_label_and_legend():
    hits, clusters, truth, detector = _one_pixel_event()
    fig = plot_event(hits, clusters, truth, detector, event_id=0, theme=PRINT)
    ax = fig.axes[0]
    assert ax.get_title() != ""
    assert ax.get_xlabel() == "x [um]"
    assert ax.get_ylabel() == "y [um]"
    assert ax.get_legend() is not None
    colorbar_ax = fig.axes[1]
    assert colorbar_ax.get_ylabel() == "charge"


def test_plot_event_present_drops_title_axes_and_legend_but_keeps_colorbar():
    hits, clusters, truth, detector = _one_pixel_event()
    fig = plot_event(hits, clusters, truth, detector, event_id=0, theme=PRESENT)
    ax = fig.axes[0]
    assert ax.get_title() == ""
    assert list(ax.get_xticks()) == []
    assert list(ax.get_yticks()) == []
    assert all(not spine.get_visible() for spine in ax.spines.values())
    assert ax.get_legend() is None
    # the colorbar is quantitative content, not a coordinate axis -- it
    # stays, but its "charge" text label (chrome) is dropped
    colorbar_ax = fig.axes[1]
    assert colorbar_ax.get_ylabel() == ""


def test_plot_event_default_theme_matches_print():
    hits, clusters, truth, detector = _one_pixel_event()
    fig = plot_event(hits, clusters, truth, detector, event_id=0)
    assert fig.axes[0].get_title() != ""


def test_plot_event_marks_entry_with_a_dot_and_exit_with_a_filled_triangle():
    hits, clusters, truth, detector = _one_pixel_event()
    fig = plot_event(hits, clusters, truth, detector, event_id=0)
    ax = fig.axes[0]
    # single-point (marker-only) lines, as opposed to the 2-point connecting
    # track line or any other multi-point line on the axes
    point_markers = [line.get_marker() for line in ax.lines if len(line.get_xdata()) == 1]
    assert ENTRY_MARKER in point_markers
    assert EXIT_MARKER in point_markers
    assert "x" not in point_markers  # the old cross marker must be gone


def test_footprint_center_is_the_pixel_footprints_geometric_midpoint():
    # pixel i spans continuous index range [i, i+1); a footprint occupying
    # ix in {10, 11} spans [10, 12), so its center is 11.0, not 10.5.
    ix = pd.Series([10, 11, 10, 11])
    iy = pd.Series([20, 20, 21, 21])
    assert _footprint_center(ix, iy) == (11.0, 21.0)


def test_footprint_center_of_a_single_pixel_is_its_own_center():
    ix = pd.Series([5])
    iy = pd.Series([9])
    assert _footprint_center(ix, iy) == (5.5, 9.5)


def test_clip_window_exact_fit_no_rounding_needed():
    # center=11.0, size=4 -> ideal_lo=9.0 exactly; window [9, 13) is
    # perfectly centered on 11.0 (window center = (9+13)/2 = 11.0)
    assert _clip_window(11.0, 4, n_max=200) == (9, 13)


def test_clip_window_half_pixel_center_still_lands_exactly():
    # center=11.5, size=5 -> ideal_lo=9.0 exactly; window [9, 14) is
    # perfectly centered on 11.5 (window center = (9+14)/2 = 11.5)
    assert _clip_window(11.5, 5, n_max=200) == (9, 14)


def test_clip_window_clips_to_the_low_edge():
    assert _clip_window(1.0, 10, n_max=200) == (0, 10)


def test_clip_window_clips_to_the_high_edge():
    assert _clip_window(198.0, 10, n_max=200) == (190, 200)


def test_clip_window_larger_than_n_max_pins_lo_to_zero():
    # size > n_max: position clips to 0, but the returned window still
    # extends `size` beyond it (pre-existing behavior, unrelated to
    # centering) -- callers never render pixels past n_pixels_x/y anyway.
    assert _clip_window(5.0, 300, n_max=200) == (0, 300)


def _asymmetric_charge_cluster(hi_charge=5.0, lo_charge=0.2):
    # the evt5 bug case: a 2x2 cluster (ix in {10, 11}, iy in {20, 21}) whose
    # charge is heavily skewed toward one corner -- a charge-weighted
    # centroid would land almost on top of that one pixel, badly off-
    # centering a window built around it. lo_charge must clear plot_event's
    # default readout_threshold (0.15) so all 4 pixels stay "hit" there too.
    detector = DetectorConfig()
    hits = pd.DataFrame(
        [
            dict(event_id=0, ix=10, iy=20, x_center_um=None, y_center_um=None, charge=hi_charge, cluster_id=0),
            dict(event_id=0, ix=10, iy=21, x_center_um=None, y_center_um=None, charge=lo_charge, cluster_id=0),
            dict(event_id=0, ix=11, iy=20, x_center_um=None, y_center_um=None, charge=lo_charge, cluster_id=0),
            dict(event_id=0, ix=11, iy=21, x_center_um=None, y_center_um=None, charge=lo_charge, cluster_id=0),
        ],
        columns=CLUSTERED_HITS_COLUMNS,
    )
    hits["x_center_um"] = (hits["ix"] + 0.5) * detector.pitch_x_um
    hits["y_center_um"] = (hits["iy"] + 0.5) * detector.pitch_y_um
    total_charge = hits["charge"].sum()
    clusters = pd.DataFrame(
        [
            dict(
                event_id=0, cluster_id=0, n_pixels=4, charge_sum=total_charge,
                x_centroid_um=(hits["x_center_um"] * hits["charge"]).sum() / total_charge,
                y_centroid_um=(hits["y_center_um"] * hits["charge"]).sum() / total_charge,
                x_centroid_digital_um=hits["x_center_um"].mean(), y_centroid_digital_um=hits["y_center_um"].mean(),
                x_span_pixels=2, y_span_pixels=2,
            )
        ],
        columns=CLUSTERS_COLUMNS,
    )
    return hits, clusters, detector


def test_zoom_pixel_bounds_centers_on_footprint_not_charge_weighted_centroid():
    hits, clusters, detector = _asymmetric_charge_cluster()
    ix_lo, ix_hi, iy_lo, iy_hi = _zoom_pixel_bounds(hits, clusters, detector, zoom=(5, 4))
    # matches _clip_window(11.0, 5, 200) and _clip_window(21.0, 4, 200)
    # exactly, regardless of how lopsided the charge is within the cluster
    assert (ix_lo, ix_hi) == (8, 13)
    assert (iy_lo, iy_hi) == (19, 23)
    # the cluster {10, 11} x {20, 21} sits inside that window with as little
    # left-over asymmetry as an odd/even window size allows (margins on
    # either side differ by at most one pixel, never pinned against one edge
    # the way the pre-fix charge-weighted-centroid version could be)
    margin_before_x, margin_after_x = 10 - ix_lo, (ix_hi - 1) - 11
    margin_before_y, margin_after_y = 20 - iy_lo, (iy_hi - 1) - 21
    assert abs(margin_before_x - margin_after_x) <= 1
    assert abs(margin_before_y - margin_after_y) <= 1


def test_zoom_pixel_bounds_is_unaffected_by_how_lopsided_the_charge_is():
    # same 2x2 footprint, far more extreme charge skew -- must center
    # identically, since centering must depend only on which pixels are
    # hit, never on how much charge is in each.
    hits_mild, clusters_mild, detector = _asymmetric_charge_cluster(hi_charge=1.1, lo_charge=1.0)
    hits_extreme, clusters_extreme, _ = _asymmetric_charge_cluster(hi_charge=1000.0, lo_charge=0.001)
    bounds_mild = _zoom_pixel_bounds(hits_mild, clusters_mild, detector, zoom=(5, 4))
    bounds_extreme = _zoom_pixel_bounds(hits_extreme, clusters_extreme, detector, zoom=(5, 4))
    assert bounds_mild == bounds_extreme


def test_plot_event_zoom_centers_the_cluster_in_the_view():
    hits, clusters, detector = _asymmetric_charge_cluster()
    truth = pd.DataFrame([], columns=TRUTH_COLUMNS)
    fig = plot_event(hits, clusters, truth, detector, event_id=0, zoom=(5, 4))
    ax = fig.axes[0]
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    # window in um must match _zoom_pixel_bounds's pixel window exactly
    assert (x_lo, x_hi) == pytest.approx((8 * detector.pitch_x_um, 13 * detector.pitch_x_um))
    assert (y_lo, y_hi) == pytest.approx((19 * detector.pitch_y_um, 23 * detector.pitch_y_um))


def _two_particle_truth(detector):
    return pd.DataFrame(
        [
            dict(event_id=0, particle_id=0, x0_um=250.0, y0_um=500.0, dxdz=0.1, dydz=0.0, charge_deposited=1.0, path_length_um=150.0),
            dict(event_id=0, particle_id=1, x0_um=300.0, y0_um=550.0, dxdz=-0.05, dydz=0.02, charge_deposited=1.0, path_length_um=150.0),
        ],
        columns=TRUTH_COLUMNS,
    )


def _empty_hits():
    return pd.DataFrame([], columns=CLUSTERED_HITS_COLUMNS)


def _two_hit_pixels(low_charge=0.1, high_charge=2.0):
    # both centered well inside the default 20x20-pixel window around
    # (275, 525) (the mean of _two_particle_truth's x0_um/y0_um); low_charge
    # sits below the default readout_threshold (0.15) so it's excludable.
    return pd.DataFrame(
        [
            dict(event_id=0, ix=11, iy=10, x_center_um=275.0, y_center_um=500.0, charge=high_charge, cluster_id=0),
            dict(event_id=0, ix=11, iy=11, x_center_um=275.0, y_center_um=550.0, charge=low_charge, cluster_id=0),
        ],
        columns=CLUSTERED_HITS_COLUMNS,
    )


def _poly3d_collections(ax):
    return [c for c in ax.collections if isinstance(c, Poly3DCollection)]


def test_plot_event_3d_is_always_axis_free_regardless_of_style():
    # plot_event_3d has no theme argument at all -- it's always presenter-only.
    detector = DetectorConfig()
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_empty_hits(), truth, detector, event_id=0)
    ax = fig.axes[0]
    assert ax.axison is False


def test_plot_event_3d_has_no_colorbar():
    # "w/o palette": hit-pixel colors are computed directly, no fig.colorbar
    detector = DetectorConfig()
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_two_hit_pixels(), truth, detector, event_id=0)
    assert len(fig.axes) == 1


def test_plot_event_3d_draws_one_line_per_truth_particle():
    detector = DetectorConfig()
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_empty_hits(), truth, detector, event_id=0)
    ax = fig.axes[0]
    # the 2-point connecting track lines, as opposed to the single-point
    # entry/exit marker "lines" also on ax.lines
    track_lines = [line for line in ax.lines if len(line.get_data_3d()[0]) == 2]
    assert len(track_lines) == 2


def test_plot_event_3d_marks_entry_with_a_dot_and_exit_with_a_filled_triangle():
    detector = DetectorConfig(thickness_um=150.0)
    truth = _two_particle_truth(detector)  # particle 0: x0=250, y0=500, dxdz=0.1, dydz=0.0
    fig = plot_event_3d(_empty_hits(), truth, detector, event_id=0)
    ax = fig.axes[0]
    point_markers = [line for line in ax.lines if len(line.get_data_3d()[0]) == 1]

    entry_points = [line for line in point_markers if line.get_marker() == ENTRY_MARKER]
    exit_points = [line for line in point_markers if line.get_marker() == EXIT_MARKER]
    assert len(entry_points) == 2  # one per truth particle
    assert len(exit_points) == 2

    # particle 0's entry sits at (x0, y0, z=0); its exit at
    # (x0 + thickness*dxdz, y0 + thickness*dydz, z=thickness)
    def _point(line):
        x, y, z = line.get_data_3d()
        return float(x[0]), float(y[0]), float(z[0])

    entry_xyz = [_point(line) for line in entry_points]
    exit_xyz = [_point(line) for line in exit_points]
    assert any(p == pytest.approx((250.0, 500.0, 0.0)) for p in entry_xyz)
    assert any(p == pytest.approx((265.0, 500.0, 150.0)) for p in exit_xyz)


def test_plot_event_3d_box_spans_full_thickness():
    detector = DetectorConfig(thickness_um=200.0)
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_empty_hits(), truth, detector, event_id=0)
    ax = fig.axes[0]
    z_lo, z_hi = ax.get_zlim()
    assert z_lo == 0.0
    assert z_hi == 200.0


def test_plot_event_3d_default_zoom_uses_20x20_pixel_window():
    detector = DetectorConfig()  # pitch_x_um=25, pitch_y_um=50
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_empty_hits(), truth, detector, event_id=0)
    ax = fig.axes[0]
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    assert x_hi - x_lo == 20 * 25.0
    assert y_hi - y_lo == 20 * 50.0


def test_plot_event_3d_zoom_overrides_default():
    detector = DetectorConfig()
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_empty_hits(), truth, detector, event_id=0, zoom=(4, 6))
    ax = fig.axes[0]
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    assert x_hi - x_lo == 4 * 25.0
    assert y_hi - y_lo == 6 * 50.0


def test_plot_event_3d_zoom_centers_on_hit_footprint_not_truth_entry_points():
    detector = DetectorConfig()  # pitch_x_um=25, pitch_y_um=50
    # truth entry points average to (275, 525) -- must NOT be used once hits
    # exist, same fix as plot_event's --zoom.
    truth = _two_particle_truth(detector)
    # only the high-charge pixel (ix=11, iy=10) clears the default
    # readout_threshold of 0.15
    fig = plot_event_3d(_two_hit_pixels(), truth, detector, event_id=0)
    ax = fig.axes[0]
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    expected_cx = (11 + 11 + 1) / 2.0 * detector.pitch_x_um  # 287.5
    expected_cy = (10 + 10 + 1) / 2.0 * detector.pitch_y_um  # 525.0
    assert (x_lo, x_hi) == pytest.approx((expected_cx - 10 * 25.0, expected_cx + 10 * 25.0))
    assert (y_lo, y_hi) == pytest.approx((expected_cy - 10 * 50.0, expected_cy + 10 * 50.0))


def test_plot_event_3d_zoom_falls_back_to_truth_entry_points_with_no_hits():
    detector = DetectorConfig()
    truth = _two_particle_truth(detector)  # x0 mean=275, y0 mean=525
    fig = plot_event_3d(_empty_hits(), truth, detector, event_id=0)
    ax = fig.axes[0]
    x_lo, x_hi = ax.get_xlim()
    y_lo, y_hi = ax.get_ylim()
    assert (x_lo, x_hi) == pytest.approx((275.0 - 10 * 25.0, 275.0 + 10 * 25.0))
    assert (y_lo, y_hi) == pytest.approx((525.0 - 10 * 50.0, 525.0 + 10 * 50.0))


def test_plot_event_3d_raises_for_event_with_no_truth():
    detector = DetectorConfig()
    truth = _two_particle_truth(detector)
    with pytest.raises(ValueError):
        plot_event_3d(_empty_hits(), truth, detector, event_id=99)


def test_plot_event_3d_draws_a_darker_pixel_grid_on_the_top_surface():
    detector = DetectorConfig(thickness_um=150.0)
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_empty_hits(), truth, detector, event_id=0)
    ax = fig.axes[0]
    grids = [c for c in ax.collections if isinstance(c, Line3DCollection)]
    assert len(grids) == 1
    segments = grids[0]._segments3d
    assert len(segments) > 0
    # every grid segment lies exactly on the readout (top) surface, z=thickness_um
    for segment in segments:
        for point in segment:
            assert point[2] == pytest.approx(150.0)
    # darker than a barely-there grid, but still translucent (not the opaque box edges)
    alpha = grids[0].get_colors()[0][3]
    assert 0.5 < alpha < 1.0


def test_plot_event_3d_sensor_box_fill_is_light():
    detector = DetectorConfig()
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_empty_hits(), truth, detector, event_id=0)
    ax = fig.axes[0]
    box = _poly3d_collections(ax)[0]  # box is added first
    face_alpha = box.get_facecolor()[0][3]
    assert face_alpha < 0.1  # lightly translucent fill, distinct from the more opaque edges
    edge_alpha = box.get_edgecolor()[0][3]
    assert edge_alpha > face_alpha  # only the fill was lightened, not the edges


def test_plot_event_3d_marks_only_pixels_above_readout_threshold():
    detector = DetectorConfig()
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_two_hit_pixels(), truth, detector, event_id=0)  # default readout_threshold=0.15
    ax = fig.axes[0]
    hit_layers = _poly3d_collections(ax)
    assert len(hit_layers) == 2  # the sensor box is a Poly3DCollection too; the hit layer is added last
    # 6 faces (a full 3D voxel) for the one pixel that clears the threshold
    assert len(hit_layers[-1].get_facecolor()) == 6


def test_plot_event_3d_readout_threshold_can_include_the_low_charge_pixel():
    detector = DetectorConfig()
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_two_hit_pixels(), truth, detector, event_id=0, readout_threshold=0.0)
    ax = fig.axes[0]
    hit_layers = _poly3d_collections(ax)
    assert len(hit_layers[-1].get_facecolor()) == 12  # 2 pixels x 6 faces each


def test_plot_event_3d_hit_voxel_spans_the_full_thickness():
    # the "3D pixel" itself -- not just a flat square -- must span z=0 to
    # z=thickness_um, since it represents the whole silicon column traversed.
    detector = DetectorConfig(thickness_um=150.0)
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_two_hit_pixels(), truth, detector, event_id=0, readout_threshold=0.0)
    ax = fig.axes[0]
    hit_layer = _poly3d_collections(ax)[-1]
    zs = hit_layer._faces[..., 2]
    assert zs.min() == pytest.approx(0.0)
    assert zs.max() == pytest.approx(150.0)


def test_plot_event_3d_hit_pixel_colors_come_from_the_charge_colormap():
    detector = DetectorConfig()
    truth = _two_particle_truth(detector)
    fig = plot_event_3d(_two_hit_pixels(high_charge=2.0), truth, detector, event_id=0, readout_threshold=0.0)
    ax = fig.axes[0]
    hit_layer = _poly3d_collections(ax)[-1]
    cmap = plt.get_cmap(CHARGE_CMAP)
    # the high-charge pixel (normalized to 1.0, since it's the event's max) should be the colormap's top color
    expected_top = mcolors.to_rgba(cmap(1.0))
    # 6 faces share that pixel's color, so it should appear (at least) 6 times
    matches = [c for c in hit_layer.get_facecolor() if tuple(c) == expected_top]
    assert len(matches) == 6
