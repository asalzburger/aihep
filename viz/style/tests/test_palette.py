from __future__ import annotations

import subprocess
import sys

from viz_style import palette


def test_on_aliases_first_categorical_color():
    assert palette.ON == palette.CATEGORICAL_OKABE_ITO[0]


def test_categorical_palette_has_six_colors():
    assert len(palette.CATEGORICAL_OKABE_ITO) == 6


def test_every_color_constant_is_a_plain_string():
    for name in (
        "TRUTH", "CLUSTER", "CLUSTER_DIGITAL", "DIGITAL_ON", "GRID", "LAYER",
        "HIT", "VERTEX", "EDGE", "TRUE_EDGE", "ON",
        "SEQUENTIAL_CHARGE_CMAP", "SEQUENTIAL_CONFUSION_CMAP",
    ):
        assert isinstance(getattr(palette, name), str)
    for hue in palette.CATEGORICAL_OKABE_ITO:
        assert isinstance(hue, str)
    for hue in palette.CLASS_COLORS:
        assert isinstance(hue, str)


def test_palette_module_never_imports_matplotlib():
    # Run in a fresh subprocess -- other test modules in this session may
    # have already imported matplotlib, which would pollute a same-process
    # sys.modules check regardless of test order.
    result = subprocess.run(
        [sys.executable, "-c", "import viz_style; import sys; assert 'matplotlib' not in sys.modules"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
