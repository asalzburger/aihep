import math

import pandas as pd
from detector2d.geometry import LineLayer

from tracker.digitize import digitize_hits


def _hit(layer_id, s_local, hit_id=0, event_id=0, particle_id=0):
    return dict(
        event_id=event_id,
        particle_id=particle_id,
        layer_id=layer_id,
        hit_id=hit_id,
        x=0.0,
        y=0.0,
        s_local=s_local,
        path_length=1.0,
    )


def test_digitize_floors_s_local_by_pitch():
    hits = pd.DataFrame([_hit(0, 0.5), _hit(0, 2.4), _hit(0, 2.6)])
    layers = [LineLayer(layer_id=0, p1=(0, 0), p2=(1, 0), pitch=1.0)]
    out = digitize_hits(hits, layers)
    assert list(out["cell_index"]) == [0.0, 2.0, 2.0]


def test_digitize_leaves_layers_without_pitch_as_nan():
    hits = pd.DataFrame([_hit(1, 3.7)])
    layers = [LineLayer(layer_id=1, p1=(0, 0), p2=(1, 0), pitch=None)]
    out = digitize_hits(hits, layers)
    assert math.isnan(out["cell_index"].iloc[0])
