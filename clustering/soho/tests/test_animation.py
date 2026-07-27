"""05_animation.py needs cairosvg + pillow (and ffmpeg on PATH to actually
encode a video, which these tests don't attempt - rendering+encoding the
full animation is slow and produces large binary output, not a fast unit
test). These tests only cover the pure helper functions, and skip cleanly
if the optional imaging deps aren't installed."""

import pytest

try:
    import cairosvg  # noqa: F401
    from PIL import Image  # noqa: F401
except (ImportError, OSError) as exc:  # pragma: no cover - environment dependent
    pytest.skip(f"cairosvg/pillow not usable: {exc}", allow_module_level=True)

from conftest import PYTHON_DIR, load_module

animation = load_module(PYTHON_DIR / "05_animation.py", "animation_mod")


def test_ease_in_out_is_smoothstep():
    assert animation.ease_in_out(0.0) == pytest.approx(0.0)
    assert animation.ease_in_out(1.0) == pytest.approx(1.0)
    assert animation.ease_in_out(0.5) == pytest.approx(0.5)
    # zero derivative at the endpoints -> flatter than linear near them
    assert animation.ease_in_out(0.1) < 0.1
    assert animation.ease_in_out(0.9) > 0.9


def test_pump_native_within_map_bounds():
    x, y = animation.PUMP_NATIVE
    assert 0 <= x <= animation.WIDTH
    assert 0 <= y <= animation.HEIGHT


def test_build_clean_base_svg_is_well_formed(monkeypatch):
    # BASE_MAP_PATH is relative to python/ (how the script is meant to be run).
    monkeypatch.chdir(PYTHON_DIR)

    svg = animation.build_clean_base_svg()

    assert svg.startswith("<?xml")
    assert svg.strip().endswith("</svg>")
    assert f'width="{animation.WIDTH}"' in svg


def test_render_base_png_matches_requested_width(monkeypatch):
    monkeypatch.chdir(PYTHON_DIR)

    image = animation.render_base_png(hires_w=200)

    assert image.size[0] == 200
    assert image.mode == "RGB"
