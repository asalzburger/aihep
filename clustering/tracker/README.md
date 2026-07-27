# tracker

Turns `tracksim2d` hits into clusters: adjacent-cell grouping along each
detector layer, the tracking-detector analogue of
[`sensor`](../sensor)'s pixel-grid clustering. Deliberately
1D and dependency-light (no scipy) -- unlike a 2D pixel grid, a single
layer's hit cells are just a sorted sequence, so "adjacent" is a gap test
between consecutive sorted cell indices.

## Project layout

| module | contents |
|---|---|
| `tracker.edm` | `HITS_COLUMNS` (a digitized hit: the `tracksim2d` hit columns + `cell_index`), `CLUSTERS_COLUMNS`. |
| `tracker.digitize` | `digitize_hits(hits_df, layers)`: `cell_index = floor(s_local / layer.pitch)`. Layers with no `pitch` are left un-digitized (`cell_index = NaN`). |
| `tracker.clustering` | `cluster_hits(hits_df, connectivity_gap=1)`: 1D connected components of cell indices, per `(event_id, layer_id)`. |
| `tracker.io` | Read/write digitized `hits`/`clusters` tables as CSV or Apache Arrow (the serialization itself lives in [`clustering/utils`](../utils), shared with `clustering/sensor`). |

## Setup

```bash
cd clustering/tracker
python3 -m venv .venv
.venv/bin/pip install -e ../../detector/detector2d -e ../utils -e . -r requirements.txt
```

(`detector2d` and `clustering_utils` are sibling path packages, not on
PyPI, so they're installed explicitly rather than listed as formal
dependencies.)

## Using it as a library

```python
from tracksim2d.simulate import simulate_events
from tracker.digitize import digitize_hits
from tracker.clustering import cluster_hits

particles, hits = simulate_events(config)          # from tracksim2d
digitized = digitize_hits(hits, config.layers)      # adds cell_index
clustered_hits, clusters = cluster_hits(digitized)  # adds cluster_id, aggregates
```

## Output schema (event data model)

Defined once in `tracker/edm.py`. Joined by `event_id` (and
`clusters`/`hits` also by `layer_id`):

**`hits`** (digitized): `event_id, particle_id, layer_id, hit_id, x, y, s_local, path_length, cell_index, cluster_id`
(`cluster_id = -1` for hits on a layer with no `pitch`, which can't be
digitized/clustered)

**`clusters`**: `event_id, layer_id, cluster_id, n_cells, n_hits, s_centroid, x_centroid, y_centroid`

`cluster_id` is numbered uniquely within an event, across all of that
event's layers (mirroring how `sensor` numbers clusters per event
across its whole pixel grid).

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers digitization (flooring, un-pitched layers), clustering (adjacent
cells merge, a gap splits clusters, cluster ids are unique per event,
un-digitized hits are excluded, centroid computation), and CSV/Arrow
round-trip.
