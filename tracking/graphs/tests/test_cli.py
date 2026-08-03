import math

import pandas as pd
from detectorsim2d.edm import HITS_COLUMNS, PARTICLES_COLUMNS
from detectorsim2d.io import write_run

from graphs.cli import main


def _write_fake_run(run_dir, fmt="csv"):
    particles = pd.DataFrame(
        [dict(event_id=0, particle_id=0, x0=0.0, y0=0.0, phi0=0.0, charge=1.0, radius=math.nan)],
        columns=PARTICLES_COLUMNS,
    )
    hits = pd.DataFrame(
        [
            dict(event_id=0, particle_id=0, layer_id=0, hit_id=0, x=10.0, y=0.0, s_local=5.0, path_length=10.0),
            dict(event_id=0, particle_id=0, layer_id=1, hit_id=1, x=20.0, y=0.0, s_local=5.0, path_length=20.0),
        ],
        columns=HITS_COLUMNS,
    )
    write_run(run_dir, fmt, particles, hits)


def test_build_writes_edges_file(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_fake_run(run_dir)
    output_dir = tmp_path / "graph_out"

    main(["build", "--run-dir", str(run_dir), "--format", "csv", "--output-dir", str(output_dir)])

    edges = pd.read_csv(output_dir / "edges.csv")
    assert len(edges) == 1  # C(2, 2) hits -> 1 fully-connected edge
    out = capsys.readouterr().out
    assert "Built 1 edge(s)" in out


def test_build_with_label_truth_adds_column_and_prints_purity(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_fake_run(run_dir)  # both hits share particle_id=0 -> the one edge is true
    output_dir = tmp_path / "graph_out"

    main(
        ["build", "--run-dir", str(run_dir), "--format", "csv", "--output-dir", str(output_dir), "--label-truth"]
    )

    edges = pd.read_csv(output_dir / "edges.csv")
    assert "is_true_edge" in edges.columns
    assert bool(edges.iloc[0]["is_true_edge"]) is True
    out = capsys.readouterr().out
    assert "purity: 1.000" in out


def test_visualize_saves_a_figure(tmp_path):
    run_dir = tmp_path / "run"
    _write_fake_run(run_dir)
    save_path = tmp_path / "graph0.png"

    main(
        [
            "visualize",
            "--run-dir",
            str(run_dir),
            "--format",
            "csv",
            "--event-id",
            "0",
            "--save",
            str(save_path),
        ]
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0


def test_visualize_with_label_truth_saves_a_figure(tmp_path):
    run_dir = tmp_path / "run"
    _write_fake_run(run_dir)
    save_path = tmp_path / "graph0_truth.png"

    main(
        [
            "visualize",
            "--run-dir",
            str(run_dir),
            "--format",
            "csv",
            "--event-id",
            "0",
            "--label-truth",
            "--save",
            str(save_path),
        ]
    )

    assert save_path.exists()
    assert save_path.stat().st_size > 0
