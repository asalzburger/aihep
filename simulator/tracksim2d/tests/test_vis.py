import math

import pandas as pd
import pytest
from detector2d.calorimeter import build_calo_stack
from detector2d.field import FieldRegion, FieldRegions
from detector2d.geometry import CircleLayer, LineLayer, Trajectory
from detector2d.polygon import build_muon_system
from detector2d.propagate import propagate
from viz_style import PRESENT, PRINT

from tracksim2d.edm import DEPOSITS_COLUMNS, HITS_COLUMNS, PARTICLES_COLUMNS
from tracksim2d.vis import _arc_path_d, _track_end_s, export_svg, plot_event, plot_lego
from tracksim2d.simulate import boundary_crossing_s, trajectory_for_row


def _one_particle_event():
    particles = pd.DataFrame(
        [dict(event_id=0, particle_id=0, x0=0.0, y0=0.0, phi0=0.0, charge=1.0, radius=math.nan)],
        columns=PARTICLES_COLUMNS,
    )
    hits = pd.DataFrame(
        [dict(event_id=0, particle_id=0, layer_id=0, hit_id=0, x=10.0, y=0.0, s_local=5.0, path_length=10.0)],
        columns=HITS_COLUMNS,
    )
    layers = [LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0))]
    return particles, hits, layers


def test_plot_event_print_keeps_title_axes_and_legend():
    particles, hits, layers = _one_particle_event()
    fig = plot_event(particles, hits, layers, event_id=0, theme=PRINT)
    ax = fig.axes[0]
    assert ax.get_title() != ""
    assert ax.get_xlabel() == "x"
    assert ax.get_ylabel() == "y"
    assert ax.get_legend() is not None


def test_plot_event_present_drops_title_axes_and_legend():
    particles, hits, layers = _one_particle_event()
    fig = plot_event(particles, hits, layers, event_id=0, theme=PRESENT)
    ax = fig.axes[0]
    assert ax.get_title() == ""
    assert list(ax.get_xticks()) == []
    assert list(ax.get_yticks()) == []
    assert all(not spine.get_visible() for spine in ax.spines.values())
    assert ax.get_legend() is None


def test_plot_event_default_theme_matches_print():
    particles, hits, layers = _one_particle_event()
    fig = plot_event(particles, hits, layers, event_id=0)
    assert fig.axes[0].get_title() != ""


def test_arc_path_d_straight_track_is_a_line():
    row = pd.Series(dict(x0=0.0, y0=0.0, phi0=0.0, radius=math.nan))
    trajectory = trajectory_for_row(row)
    d = _arc_path_d(trajectory, 10.0)
    assert d == "M 0.000,0.000 L 10.000,0.000"


def test_arc_path_d_curved_track_uses_arc_commands_and_ends_at_expected_point():
    row = pd.Series(dict(x0=0.0, y0=0.0, phi0=0.0, radius=10.0))
    trajectory = trajectory_for_row(row)
    s_end = math.pi * 10.0 / 2.0  # quarter turn
    d = _arc_path_d(trajectory, s_end)
    assert d.startswith("M 0.000,0.000")
    assert " A " in d
    assert "10.000,10.000" in d  # the known quarter-turn endpoint


def test_arc_path_d_handles_spans_longer_than_a_full_turn():
    row = pd.Series(dict(x0=0.0, y0=0.0, phi0=0.0, radius=5.0))
    trajectory = trajectory_for_row(row)
    d = _arc_path_d(trajectory, 3 * math.pi * 5.0)  # 1.5 full turns
    assert d.count(" A ") >= 2  # must be chunked, each span <= pi radians


def test_track_end_s_caps_farthest_hit_at_the_boundary_instead_of_looping_back():
    # a curved trajectory starting at the origin always crosses a concentric
    # circle twice (see detector2d.barrel); a hit path_length beyond the
    # boundary crossing must be capped there, not drawn out to the hit.
    # radius=10 reaches out to 2*radius=20 from the origin, so a boundary of
    # 15 is reachable.
    trajectory = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=10.0)
    far_hit = pd.DataFrame([{"path_length": 1000.0}])
    boundary_s = boundary_crossing_s(trajectory, 15.0)
    assert boundary_s is not None
    s_end = _track_end_s(trajectory, far_hit, track_length=100.0, tracker_boundary=15.0)
    assert s_end == pytest.approx(boundary_s)
    assert s_end < 1000.0


def test_track_end_s_uses_farthest_hit_when_no_boundary_set():
    trajectory = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=None)
    hit = pd.DataFrame([{"path_length": 42.0}])
    assert _track_end_s(trajectory, hit, track_length=100.0, tracker_boundary=None) == 42.0


def test_track_end_s_falls_back_to_track_length_with_no_hits():
    trajectory = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=None)
    no_hits = pd.DataFrame([], columns=["path_length"])
    assert _track_end_s(trajectory, no_hits, track_length=100.0, tracker_boundary=None) == 100.0


def test_export_svg_with_tracker_boundary_shortens_looping_arc(tmp_path):
    particles = pd.DataFrame(
        [dict(event_id=0, particle_id=0, x0=0.0, y0=0.0, phi0=0.0, charge=1.0, radius=10.0)],
        columns=PARTICLES_COLUMNS,
    )
    # no hits recorded -- without a boundary this would draw out to default_track_length
    hits = pd.DataFrame([], columns=HITS_COLUMNS)
    layers = [CircleLayer(layer_id=0, center=(0, 0), radius=8.0)]

    out = tmp_path / "event.svg"
    export_svg(
        layers, particles, hits, out, width=100, height=100, default_track_length=1000.0, tracker_boundary=15.0
    )
    d_attr = out.read_text().split('<path d="')[1].split('"')[0]
    # every arc endpoint must lie at or inside the boundary radius (15), not
    # out near the unbounded default_track_length of 1000
    coords = d_attr.replace("M ", "").replace(" A 10.000,10.000 0 0,1 ", ",").split(",")
    xs = [float(v) for v in coords[0::2]]
    ys = [float(v) for v in coords[1::2]]
    # SVG coordinates are rounded to 3 decimals, so allow for that rounding
    assert all(math.hypot(x, y) <= 15.0 + 1e-2 for x, y in zip(xs, ys))


def test_export_svg_draws_line_sensors_solid_and_circle_surfaces_dashed(tmp_path):
    particles = pd.DataFrame([], columns=PARTICLES_COLUMNS)
    hits = pd.DataFrame([], columns=HITS_COLUMNS)
    layers = [
        LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0)),  # a detailed-mode module (real sensor)
        CircleLayer(layer_id=1, center=(0, 0), radius=3.0),  # a simplified-mode bare layer surface
    ]

    out = tmp_path / "event.svg"
    export_svg(layers, particles, hits, out, width=100, height=100)
    content = out.read_text()

    line_elem = content[content.index("<line") : content.index("/>", content.index("<line")) + 2]
    circle_elem = content[content.index("<circle") : content.index("/>", content.index("<circle")) + 2]
    assert "stroke-dasharray" not in line_elem  # individual sensor: solid
    assert 'stroke-dasharray="6,3"' in circle_elem  # bare layer surface: dashed
    assert 'stroke-dasharray' not in content[: content.index("<line")]  # not set at the <g> level either


def _calo_event():
    """A photon that showered into two ECAL cells and one HCAL cell."""
    ecal = build_calo_stack(100, 210.0, 3, 30.0, 64, system="ecal", phi_stagger=[0.0, 0.5, 0.0])
    hcal = build_calo_stack(200, 300.0, 2, 80.0, 32, system="hcal")
    layers = ecal + hcal
    deposits = pd.DataFrame(
        [
            dict(event_id=0, particle_id=0, system="ecal", layer_id=100, cell_id=8,
                 x=ecal[0].cell_position(8)[0], y=ecal[0].cell_position(8)[1],
                 s_local=ecal[0].cell_local_coord(8), energy=60.0),
            dict(event_id=0, particle_id=0, system="ecal", layer_id=101, cell_id=8,
                 x=ecal[1].cell_position(8)[0], y=ecal[1].cell_position(8)[1],
                 s_local=ecal[1].cell_local_coord(8), energy=30.0),
            dict(event_id=0, particle_id=0, system="hcal", layer_id=200, cell_id=4,
                 x=hcal[0].cell_position(4)[0], y=hcal[0].cell_position(4)[1],
                 s_local=hcal[0].cell_local_coord(4), energy=10.0),
        ],
        columns=DEPOSITS_COLUMNS,
    )
    return layers, deposits


def test_plot_event_draws_a_wedge_per_deposit_on_top_of_the_calo_rings():
    layers, deposits = _calo_event()
    particles = pd.DataFrame([], columns=PARTICLES_COLUMNS)
    hits = pd.DataFrame([], columns=HITS_COLUMNS)

    without = plot_event(particles, hits, layers, event_id=0)
    with_deposits = plot_event(particles, hits, layers, event_id=0, deposits=deposits)

    extra = len(with_deposits.axes[0].patches) - len(without.axes[0].patches)
    assert extra == len(deposits)


def test_plot_event_draws_a_segmented_trajectory_through_every_field_region():
    """With a field map the drawn curve must follow the piecewise path, not
    the single arc the stored radius describes."""
    layers = build_muon_system(300, 520.0, 100.0, n_stations=3, n_planes=3)
    field = FieldRegions(
        regions=(FieldRegion(210.0, 2.0), FieldRegion(480.0, 0.0), FieldRegion(None, -1.0)),
        k=0.2998,
    )
    particles = pd.DataFrame(
        [dict(event_id=0, particle_id=0, species="mu+", pdg=-13, x0=0.0, y0=0.0, phi0=0.3,
              charge=1.0, energy=200.0, radius=math.nan)],
        columns=PARTICLES_COLUMNS,
    )
    hits = pd.DataFrame(
        [dict(event_id=0, particle_id=0, system="muon", layer_id=320, hit_id=0,
              x=700.0, y=200.0, s_local=1.0, path_length=730.0)],
        columns=HITS_COLUMNS,
    )
    fig = plot_event(
        particles, hits, layers, event_id=0, field=field, world_radius=800.0, max_path_length=4000.0
    )
    # pick the track out by its label -- the other lines are the 72 muon
    # chamber sides and the vertex marker
    track_line = next(line for line in fig.axes[0].lines if line.get_label().startswith("mu+"))
    xs, ys = track_line.get_data()
    radii = [math.hypot(x, y) for x, y in zip(xs, ys)]
    assert max(radii) > 480.0  # really reached the muon system
    assert radii == sorted(radii)  # and went outward the whole way


def test_plot_lego_has_one_row_per_sampling_layer():
    layers, deposits = _calo_event()
    fig = plot_lego(deposits, layers, event_id=0)
    ax = fig.axes[0]
    assert [t.get_text() for t in ax.get_yticklabels()] == [
        "ecal L100", "ecal L101", "ecal L102", "hcal L200", "hcal L201"
    ]
    assert ax.get_xlabel() == "phi [deg]"
    assert ax.get_ylim() == (0.0, 5.0)


def test_plot_lego_phi_range_zooms_in():
    layers, deposits = _calo_event()
    fig = plot_lego(deposits, layers, event_id=0, phi_range=(40.0, 60.0))
    assert fig.axes[0].get_xlim() == (40.0, 60.0)


def test_plot_lego_keeps_its_axes_in_present_theme():
    """Unlike the x/y display, the unrolled view is an analysis plot -- it is
    unreadable without its phi axis, so `present` must not strip it."""
    layers, deposits = _calo_event()
    fig = plot_lego(deposits, layers, event_id=0, theme=PRESENT)
    ax = fig.axes[0]
    assert ax.get_title() == ""  # title still suppressed
    assert ax.get_xlabel() == "phi [deg]"  # but the axes survive
    assert len(ax.get_yticklabels()) == 5


def test_plot_lego_needs_a_calorimeter():
    with pytest.raises(ValueError, match="no calorimeter rings"):
        plot_lego(pd.DataFrame([], columns=DEPOSITS_COLUMNS), [], event_id=0)


def test_arc_path_d_emits_one_run_of_commands_per_field_region():
    """A segmented path stays a single exact <path>: the straight middle
    region becomes an `L`, the two bending regions `A`rcs."""
    field = FieldRegions(
        regions=(FieldRegion(210.0, 2.0), FieldRegion(480.0, 0.0), FieldRegion(None, -1.0)),
        k=0.2998,
    )
    track = propagate(0.0, 0.0, 0.3, charge=1.0, pt=200.0, field=field, world_radius=800.0)
    d = _arc_path_d(track, track.total_length)
    assert d.startswith("M 0.000,0.000")
    assert " L " in d  # the field-free calorimeter region
    assert d.count(" A ") >= 2  # the tracker and muon regions
    # the path really ends at the world radius
    x, y = (float(v) for v in d.rsplit(" ", 1)[-1].split(","))
    assert math.hypot(x, y) == pytest.approx(800.0, abs=0.1)


def test_track_end_s_extends_to_a_deposit_when_a_neutral_particle_has_no_hits():
    """A photon leaves no hits at all, so without its deposits it would be
    drawn out to the meaningless default track length."""
    trajectory = Trajectory(x0=0.0, y0=0.0, phi0=0.0, radius=None)
    no_hits = pd.DataFrame([], columns=["path_length"])
    deposits = pd.DataFrame([{"x": 285.0, "y": 0.0}])
    s_end = _track_end_s(trajectory, no_hits, track_length=100.0, tracker_boundary=None,
                         particle_deposits=deposits)
    assert s_end == pytest.approx(285.0)


def test_export_svg_draws_a_calo_ring_as_the_slab_it_occupies(tmp_path):
    rings = build_calo_stack(100, 210.0, 1, 30.0, 64, system="ecal")
    out = tmp_path / "calo.svg"
    export_svg(rings, pd.DataFrame([], columns=PARTICLES_COLUMNS),
               pd.DataFrame([], columns=HITS_COLUMNS), out, width=100, height=100)
    content = out.read_text()
    assert 'r="210.000"' in content and 'r="240.000"' in content  # inner and outer faces
    assert "stroke-dasharray" not in content  # real hardware, not an idealized surface


def test_export_svg_writes_expected_elements(tmp_path):
    particles = pd.DataFrame(
        [dict(event_id=0, particle_id=0, x0=0.0, y0=0.0, phi0=0.0, charge=1.0, radius=math.nan)],
        columns=PARTICLES_COLUMNS,
    )
    hits = pd.DataFrame(
        [dict(event_id=0, particle_id=0, layer_id=0, hit_id=0, x=10.0, y=0.0, s_local=5.0, path_length=10.0)],
        columns=HITS_COLUMNS,
    )
    layers = [LineLayer(layer_id=0, p1=(10.0, -5.0), p2=(10.0, 5.0)), CircleLayer(layer_id=1, center=(0, 0), radius=3)]

    out = tmp_path / "event.svg"
    export_svg(layers, particles, hits, out, width=100, height=100)

    content = out.read_text()
    assert content.startswith("<?xml")
    assert 'viewBox="0.0 0.0 100 100"' in content
    assert content.count("<line") == 1  # the LineLayer
    assert content.count("<circle") == 3  # CircleLayer + one hit marker + one vertex marker
    assert 'id="hits"' in content and 'cx="10.000" cy="0.000"' in content  # the hit marker
    assert 'id="vertices"' in content and 'cx="0.000" cy="0.000"' in content  # the particle's vertex
    assert "<path" in content  # the straight track
