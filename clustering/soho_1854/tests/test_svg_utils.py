import pytest
import svg_utils


def test_load_dots(tmp_path):
    csv_path = tmp_path / "dots.csv"
    csv_path.write_text("x,y\n1.0,2.0\n3.5,4.5\n")

    pts = svg_utils.load_dots(str(csv_path))

    assert pts == [(1.0, 2.0), (3.5, 4.5)]


def test_make_svg_to_latlon_reproduces_affine_transform(tmp_path):
    # A known affine map (x, y) -> (lat, long); make_svg_to_latlon should
    # recover it exactly from 3 reference points and predict correctly at a
    # 4th, non-reference point.
    def true_lat(x, y):
        return 0.001 * x - 0.0005 * y + 51.5

    def true_long(x, y):
        return -0.0003 * x + 0.0008 * y - 0.1

    ref_points = [(0.0, 0.0), (1000.0, 200.0), (300.0, 900.0)]
    csv_path = tmp_path / "gauge.csv"
    lines = ["x,y,lat,long"]
    for x, y in ref_points:
        lines.append(f"{x},{y},{true_lat(x, y)},{true_long(x, y)}")
    csv_path.write_text("\n".join(lines))

    svg_to_latlon = svg_utils.make_svg_to_latlon(str(csv_path))

    # A point not used to fit the transform.
    x, y = 542.0, 371.0
    lat, long_ = svg_to_latlon(x, y)
    assert lat == pytest.approx(true_lat(x, y))
    assert long_ == pytest.approx(true_long(x, y))


SYNTHETIC_SVG = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="10" height="10">'
    '<path d="M0 0 L1 1"/><circle cx="5" cy="5" r="2"/>'
    "</svg>"
)


def test_base_map_paths_extracts_inner_svg_content(tmp_path):
    svg_path = tmp_path / "map.svg"
    svg_path.write_text(SYNTHETIC_SVG)

    inner = svg_utils._base_map_paths(str(svg_path))

    assert inner == '<path d="M0 0 L1 1"/><circle cx="5" cy="5" r="2"/>'


def test_render_cluster_svg_writes_expected_elements(tmp_path):
    base_map = tmp_path / "base.svg"
    base_map.write_text(SYNTHETIC_SVG)
    out_path = tmp_path / "out.svg"

    svg_utils.render_cluster_svg(
        out_path=str(out_path),
        points=[(1.0, 1.0), (2.0, 2.0)],
        labels=[0, 1],
        centers=[(1.5, 1.5)],
        title="test title",
        base_map_path=str(base_map),
    )

    content = out_path.read_text()
    assert content.startswith("<?xml")
    assert "<path d=\"M0 0 L1 1\"/>" in content  # base map carried through
    assert content.count("<circle") >= 2  # one per point (centers use markers, not circles)
    assert svg_utils.PALETTE[0] in content  # cluster 0's color
    assert svg_utils.PALETTE[1] in content  # cluster 1's color
    assert "test title" in content


def test_render_cluster_svg_colors_noise_points_gray(tmp_path):
    base_map = tmp_path / "base.svg"
    base_map.write_text(SYNTHETIC_SVG)
    out_path = tmp_path / "out.svg"

    svg_utils.render_cluster_svg(
        out_path=str(out_path),
        points=[(1.0, 1.0)],
        labels=[-1],
        centers=[],
        base_map_path=str(base_map),
    )

    content = out_path.read_text()
    assert svg_utils.NOISE_COLOR in content
