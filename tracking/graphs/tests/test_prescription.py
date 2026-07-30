import pytest

from graphs.prescription import ConnectionRules, FullyConnected, Regional, parse_prescription


def test_parse_fully_connected_defaults_and_directed():
    assert parse_prescription({"kind": "fully_connected"}) == FullyConnected(directed=False)
    assert parse_prescription({"kind": "fully_connected", "directed": True}) == FullyConnected(directed=True)


def test_parse_regional():
    assert parse_prescription({"kind": "regional", "phi_width": 0.5}) == Regional(phi_width=0.5)


def test_parse_connection_rules_partial_and_full():
    partial = parse_prescription({"kind": "connection_rules", "delta_layer_id": [1, 2]})
    assert partial == ConnectionRules(delta_layer_id=(1.0, 2.0))

    full = parse_prescription(
        {
            "kind": "connection_rules",
            "delta_layer_id": [1, 2],
            "delta_r": [0.0, 100.0],
            "delta_x": [-5.0, 5.0],
            "delta_phi": [-0.3, 0.3],
        }
    )
    assert full == ConnectionRules(
        delta_layer_id=(1.0, 2.0), delta_r=(0.0, 100.0), delta_x=(-5.0, 5.0), delta_phi=(-0.3, 0.3)
    )


def test_parse_prescription_rejects_unknown_kind():
    with pytest.raises(ValueError):
        parse_prescription({"kind": "bogus"})
