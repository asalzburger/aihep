import math

import pandas as pd
from detector2d.geometry import CircleLayer, LineLayer

from tracksim2d.edm import HITS_COLUMNS, PARTICLES_COLUMNS
from tracksim2d.vis import _arc_path_d, export_svg
from tracksim2d.simulate import trajectory_for_row


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
