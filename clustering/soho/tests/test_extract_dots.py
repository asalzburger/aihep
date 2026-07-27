from conftest import PYTHON_DIR, RESOURCES_DIR, load_module

extract_dots = load_module(PYTHON_DIR / "01_extract_dots.py", "extract_dots_mod")


def test_extract_dot_centers_on_synthetic_svg(tmp_path):
    svg = (
        '<?xml version="1.0"?><svg width="10" height="10">'
        '<path style="fill:#e8000d;stroke:none" d="M0 0 L4 0 L4 4 L0 4 Z"/>'
        '<path style="fill:#000000" d="M5 5 L9 5 L9 9 L5 9 Z"/>'  # different fill, ignored
        "</svg>"
    )
    svg_path = tmp_path / "map.svg"
    svg_path.write_text(svg)

    centers = extract_dots.extract_dot_centers(str(svg_path))

    assert centers == [(2.0, 2.0)]


def test_extract_dot_centers_on_real_map_matches_committed_dots_csv():
    centers = extract_dots.extract_dot_centers(str(RESOURCES_DIR / "Soho_map_annoted.svg"))

    with open(RESOURCES_DIR / "dots.csv") as f:
        lines = f.read().strip().splitlines()[1:]  # skip header
    committed = [tuple(float(v) for v in line.split(",")) for line in lines]

    assert centers == committed
