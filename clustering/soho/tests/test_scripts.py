"""Subprocess smoke tests: run the numbered scripts exactly as documented
(`python 0N_*.py` with cwd=python/) against a throwaway copy of the project,
and check they succeed and produce the expected output files."""

import subprocess
import sys
from pathlib import Path


def run_script(python_dir: Path, script: str) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, script],
        cwd=python_dir,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"{script} failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    return result


def test_extract_dots_is_deterministic(project_copy):
    python_dir = project_copy / "python"
    original = (project_copy / "resources" / "dots.csv").read_text()

    run_script(python_dir, "01_extract_dots.py")

    regenerated = (project_copy / "resources" / "dots.csv").read_text()
    assert regenerated == original


def test_centroid_script(project_copy):
    python_dir = project_copy / "python"

    result = run_script(python_dir, "02_centroid.py")

    assert "Centroid:" in result.stdout
    assert "536 points" in result.stdout
    assert (python_dir / "out_centroid.svg").exists()


def test_kmeans_script(project_copy):
    python_dir = project_copy / "python"

    result = run_script(python_dir, "03_kmeans.py")

    for k in (1, 3, 5):
        assert f"k-means, k={k}" in result.stdout
        assert (python_dir / f"out_kmeans_k{k}.svg").exists()


def test_dbscan_script(project_copy):
    python_dir = project_copy / "python"

    result = run_script(python_dir, "04_dbscan.py")

    assert result.stdout.count("clusters=") == 3
    for tag in ("eps50_ms4", "eps80_ms5", "eps120_ms5"):
        assert (python_dir / f"out_dbscan_{tag}.svg").exists()
