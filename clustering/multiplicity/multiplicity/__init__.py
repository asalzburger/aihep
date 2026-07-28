"""Predicts a cluster's particle multiplicity (1, 2, or 3) from its
fixed-size pixel matrix: a small ordinal-regression MLP, trained on
`sensor`-shaped run directories (p1/p2/p3/p123-style single- and
multi-particle clusters) and validated on an independent one.
"""

from __future__ import annotations

from .dataset import Dataset, build_dataset, compute_matrix_shape
from .model import MAX_PARTICLES, MIN_PARTICLES, MultiplicityMLP, decode_score, encode_label
from .train import load_checkpoint, save_checkpoint, train_model

__all__ = [
    "Dataset",
    "build_dataset",
    "compute_matrix_shape",
    "MultiplicityMLP",
    "MIN_PARTICLES",
    "MAX_PARTICLES",
    "encode_label",
    "decode_score",
    "train_model",
    "save_checkpoint",
    "load_checkpoint",
]
