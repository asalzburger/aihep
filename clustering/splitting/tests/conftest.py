"""Shared test fixtures for `splitting`.

`resources/p123` is *generated* test data (via `sensor.cli run --config
configs/p123.yaml`), not a checked-in fixture -- it's regenerated fresh by
the `p123_resources` fixture before any test that needs it runs, so those
tests always exercise current `sensor` output instead of a stale or
missing directory. Tests that use it are tagged with the `p123_resources`
marker (registered below) so they can be selected/deselected on their own:

    pytest -m p123_resources          # only the tests that regenerate/use it
    pytest -m "not p123_resources"    # skip them (e.g. sensor's venv isn't set up)
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

SENSOR_DIR = Path(__file__).resolve().parent.parent.parent / "sensor"
SENSOR_PYTHON = SENSOR_DIR / ".venv" / "bin" / "python"
P123_CONFIG = SENSOR_DIR / "configs" / "p123.yaml"
RESOURCES_P123 = Path(__file__).resolve().parent.parent / "resources" / "p123"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "p123_resources: (re)generates resources/p123 via `sensor.cli run` before the test runs; "
        "needs sensor's venv set up (see clustering/sensor/README.md)",
    )


@pytest.fixture(scope="session")
def p123_resources() -> Path:
    """(Re)generate resources/p123 by running `sensor.cli run` with
    configs/p123.yaml -- once per test session, reused by every test that
    depends on it. Skips (rather than fails) if sensor's venv or config
    aren't where expected, so the rest of the suite still runs on a
    checkout that hasn't set sensor up."""
    if not SENSOR_PYTHON.exists():
        pytest.skip(f"sensor venv not set up at {SENSOR_PYTHON} -- can't regenerate resources/p123")
    if not P123_CONFIG.exists():
        pytest.skip(f"{P123_CONFIG} not found -- can't regenerate resources/p123")

    subprocess.run(
        [
            str(SENSOR_PYTHON),
            "-m",
            "sensor.cli",
            "run",
            "--config",
            str(P123_CONFIG),
            "--n-events",
            "1000",
            "--seed",
            "123",
            "--output-dir",
            str(RESOURCES_P123),
            "--format",
            "arrow",
        ],
        cwd=SENSOR_DIR,
        check=True,
        capture_output=True,
        text=True,
    )
    return RESOURCES_P123
