import pandas as pd
from viz_style import PRESENT, PRINT

from sensor.edm import CLUSTERS_COLUMNS, CLUSTERED_HITS_COLUMNS, TRUTH_COLUMNS
from sensor.sim.config import DetectorConfig
from sensor.vis import plot_event


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
