import numpy as np
import pytest
import torch

from multiplicity.model import MultiplicityMLP, decode_score, encode_label


def test_encode_label_round_trips_through_decode_score():
    n_particles = np.array([1, 2, 3])
    encoded = encode_label(n_particles)

    assert encoded == pytest.approx([0.0, 0.5, 1.0])
    assert list(decode_score(encoded)) == [1, 2, 3]


def test_encode_label_rejects_out_of_range():
    with pytest.raises(ValueError):
        encode_label(np.array([0, 1, 2]))
    with pytest.raises(ValueError):
        encode_label(np.array([1, 2, 4]))


@pytest.mark.parametrize(
    "score,expected",
    [
        (0.0, 1),
        (0.32, 1),
        (0.34, 2),
        (0.5, 2),
        (0.65, 2),
        (0.67, 3),
        (1.0, 3),
    ],
)
def test_decode_score_bin_boundaries(score, expected):
    assert decode_score(np.array([score]))[0] == expected


def test_mlp_forward_shape_and_range():
    model = MultiplicityMLP(n_x=4, n_y=2)
    x = torch.rand(10, 4, 2)

    out = model(x)

    assert out.shape == (10,)
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_mlp_hidden_layers_have_32_nodes():
    model = MultiplicityMLP(n_x=4, n_y=2)
    linear_layers = [m for m in model.net if isinstance(m, torch.nn.Linear)]

    assert len(linear_layers) == 3  # input->hidden, hidden->hidden, hidden->output
    assert linear_layers[0].out_features == 32
    assert linear_layers[1].in_features == 32
    assert linear_layers[1].out_features == 32
    assert linear_layers[2].out_features == 1
