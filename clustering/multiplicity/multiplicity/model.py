"""A small ordinal-regression MLP: two 32-node hidden layers, one sigmoid
output score in [0, 1]. Particle multiplicity (1..MAX_PARTICLES) is encoded
as an equally-spaced target on that range and decoded back via MAX_PARTICLES
equal-width bins -- e.g. for 1/2/3 particles: [0, 0.33), [0.33, 0.66),
[0.66, 1] -- rather than a 3-node softmax, since the classes have a natural
order (1 < 2 < 3 particles) that a single ordinal score respects directly.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

N_HIDDEN = 32
MIN_PARTICLES = 1
MAX_PARTICLES = 3


class MultiplicityMLP(nn.Module):
    def __init__(self, n_x: int, n_y: int, n_hidden: int = N_HIDDEN):
        super().__init__()
        self.input_shape = (n_x, n_y)
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_x * n_y, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def encode_label(
    n_particles: np.ndarray, min_particles: int = MIN_PARTICLES, max_particles: int = MAX_PARTICLES
) -> np.ndarray:
    """n_particles (min_particles..max_particles) -> regression target in
    [0, 1], equally spaced: min_particles -> 0.0, max_particles -> 1.0."""
    n_particles = np.asarray(n_particles)
    if np.any((n_particles < min_particles) | (n_particles > max_particles)):
        raise ValueError(f"n_particles must be in [{min_particles}, {max_particles}], got {n_particles}")
    return (n_particles - min_particles).astype(np.float32) / (max_particles - min_particles)


def decode_score(
    score: np.ndarray, min_particles: int = MIN_PARTICLES, max_particles: int = MAX_PARTICLES
) -> np.ndarray:
    """[0, 1] score -> predicted class in {min_particles, ..., max_particles}
    via (max_particles - min_particles + 1) equal-width bins."""
    n_classes = max_particles - min_particles + 1
    bin_index = np.clip((np.asarray(score) * n_classes).astype(int), 0, n_classes - 1)
    return bin_index + min_particles
