import torch

from flavor_tagging.model import BTaggerMLP


def test_forward_shape_and_range():
    model = BTaggerMLP(n_features=10)
    x = torch.randn(5, 10)
    out = model(x)
    assert out.shape == (5,)
    assert torch.all((out >= 0.0) & (out <= 1.0))


def test_n_features_is_recorded_on_the_model():
    model = BTaggerMLP(n_features=7)
    assert model.n_features == 7
