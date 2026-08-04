"""Smoke tests: each evaluate.py plotting function runs end to end and
produces a real file."""

import matplotlib

matplotlib.use("Agg")  # headless: these tests only check that a file gets written

import numpy as np

from flavor_tagging.evaluate import plot_confusion_matrix, plot_roc, plot_score_distribution


def test_plot_roc_writes_a_file(tmp_path):
    fpr = np.array([0.0, 0.1, 1.0])
    tpr = np.array([0.0, 0.8, 1.0])
    plot_roc(fpr, tpr, roc_auc=0.9, save_path=tmp_path / "roc.png")
    assert (tmp_path / "roc.png").exists()


def test_plot_confusion_matrix_writes_a_file(tmp_path):
    cm = np.array([[45, 5], [8, 42]])
    plot_confusion_matrix(cm, save_path=tmp_path / "cm.png")
    assert (tmp_path / "cm.png").exists()


def test_plot_score_distribution_writes_a_file(tmp_path):
    rng = np.random.default_rng(0)
    score = rng.uniform(0.0, 1.0, size=100)
    is_b_jet = (score > 0.5).astype(int)
    plot_score_distribution(score, is_b_jet, save_path=tmp_path / "score.png")
    assert (tmp_path / "score.png").exists()
