"""A small binary-classification MLP: two hidden layers, one sigmoid output
-- the probability that a jet (given `dataset.build_dataset`'s feature
vector: leading-track `d0` slots + `n_tracks`/`n_muons`/cluster-energy
summary) is a b-jet. Plain binary cross-entropy on a single ordered class
pair (light=0, b-jet=1), unlike `clustering/multiplicity`'s ordinal
1/2/3-particle encoding -- there's no third class in between to order.
"""

from __future__ import annotations

import torch
from torch import nn

N_HIDDEN = 32


class BTaggerMLP(nn.Module):
    def __init__(self, n_features: int, n_hidden: int = N_HIDDEN):
        super().__init__()
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(n_features, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
