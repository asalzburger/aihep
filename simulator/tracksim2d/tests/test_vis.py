import math

import pandas as pd
import pytest
from detector2d.geometry import CircleLayer, LineLayer, Trajectory

from tracksim2d.edm import HITS_COLUMNS, PARTICLES_COLUMNS
from tracksim2d.vis import _arc_path_d, _track_end_s, export_svg
from tracksim2d.simulate import boundary_crossing_s, trajectory_for_row


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
