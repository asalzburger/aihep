# graphs

Candidate track-graph construction on top of
[`simulator/detectorsim2d`](../../simulator/detectorsim2d)'s hits: turns a hits
table into a graph (nodes = hits, edges = candidate hit-to-hit connections)
according to a configurable, YAML-driven **prescription** -- fully
connected, regional (phi-sector) selection, or explicit per-feature
connection rules. Works on hits from *any* `detector2d` layout without
caring which -- a flat plane stack like [`tracking/denby`](../denby)'s
recovered event, or a concentric barrel like
[`simulator/detectorsim2d/configs/barrel6.yaml`](../../simulator/detectorsim2d/configs/barrel6.yaml)
-- since only `x`, `y`, `layer_id`, and `event_id` are ever used, never the
layer geometry itself.

## Project layout

| module | contents |
|---|---|
| `graphs.edm` | `EDGES_COLUMNS` (the edges table schema: `src_hit_id`/`dst_hit_id` plus derived `delta_layer_id`/`delta_r`/`delta_phi`/`delta_x`/`delta_y`/`distance`) and `TrackGraph` (`nodes` + `edges`, the graph representation). |
| `graphs.prescription` | The prescription representation: `FullyConnected`, `Regional`, `ConnectionRules` dataclasses, and `parse_prescription(dict)`. |
| `graphs.config` | `GraphConfig`/`load_config(path)` -- the same YAML load/merge pattern as `detectorsim2d.config.load_config`, for a `prescription:` section. |
| `graphs.build` | `build_edges(hits, prescription)` / `build_graph(hits, prescription)` -- the actual pair-selection + edge-feature logic per prescription kind. |
| `graphs.io` | Read/write the edges table as CSV or Apache Arrow, reusing `detectorsim2d.io`. |
| `graphs.truth` | `label_edges`/`label_graph` (ground-truth `is_true_edge` labeling) and `purity` -- see "Edge truth labeling" below. |
| `graphs.vis` | `plot_edges_on(ax, nodes, edges)` / `plot_event_with_graph(...)` -- draws edges *on top of* `detectorsim2d.vis.plot_event`, rather than duplicating its layer/trajectory/hit drawing. Colors true/false edges differently once labeled. |

`graphs.cli` wires `build` -> `visualize` into subcommands, the same split
as `detectorsim2d.cli`.

## The graph representation

```python
from graphs.edm import TrackGraph
```

A `TrackGraph` is just two `pandas.DataFrame`s:

- **`nodes`** -- exactly the `detectorsim2d` hits table it was built from,
  unmodified. A graph's nodes *are* hits.
- **`edges`** -- one row per candidate connection, `EDGES_COLUMNS`:
  `event_id, edge_id, src_hit_id, dst_hit_id, delta_layer_id, delta_r,
  delta_phi, delta_x, delta_y, distance`. `src_hit_id`/`dst_hit_id` are
  `hits.hit_id` values (globally unique across a whole hits table, see
  `detectorsim2d.simulate.hits_for_particles`); `delta_*`/`distance` are always
  `dst - src` (`delta_phi` wrapped into `(-pi, pi]`, since phi is cyclic),
  computed the same way regardless of which prescription built the edge.
  Edges are only ever formed *within* one event -- hits from different
  events never connect.

Kept as plain DataFrames rather than a graph-library object so it round-trips
through the same CSV/Arrow IO as the rest of this codebase and overlays
directly onto a `detectorsim2d` event plot.

## The prescription representation

A prescription is *how* to decide which pairs of hits become edges --
kept fully separate from what an edge is (above) and how it's evaluated
(`graphs.build`). Three kinds, in `graphs/prescription.py`:

- **`fully_connected`** -- connect every hit to every other hit in the
  event. Quadratic in hit count; fine for a small didactic event.

  ```yaml
  prescription:
    kind: fully_connected
    directed: false   # true keeps both (i, j) and (j, i) as separate edges
  ```

- **`regional`** -- partition each event's hits into `phi_width`-wide
  azimuthal sectors (bucketed by `atan2(y, x)`) and fully-connect only
  within a sector: a spatial-locality cut, no other rule. Hits in different
  sectors never connect, even if close in `(x, y)` (no wraparound stitching
  across a sector boundary -- a known simplification).

  ```yaml
  prescription:
    kind: regional
    phi_width: 0.5   # radians
  ```

- **`connection_rules`** -- connect hit `i -> j` only if every *given*
  range contains the corresponding signed delta (`dst - src`); an omitted
  range leaves that feature unconstrained. Ranges are `(min, max)`, both
  inclusive; an asymmetric range (e.g. `[1, 3]`) naturally encodes direction
  (only "outward" connects), a symmetric one (e.g. `[-3, 3]`) allows both.
  `delta_phi` is wrapped into `(-pi, pi]` before the range check.

  ```yaml
  prescription:
    kind: connection_rules
    delta_layer_id: [1, 2]     # only to the next layer or one beyond (outward only)
    delta_r: [0.0, 100.0]      # must move outward, by no more than 100 units
    delta_phi: [-0.3, 0.3]     # radians; stay within a narrow azimuthal window
    # delta_x: [-5.0, 5.0]     # omit any of the four to leave it unconstrained
  ```

See `configs/` for these three as complete, working files. Building your own
`Prescription` directly (skipping YAML) works the same way `detector2d`
layers do -- just construct the dataclass:

```python
from graphs.prescription import ConnectionRules
from graphs.build import build_graph

prescription = ConnectionRules(delta_layer_id=(1.0, 1.0))
graph = build_graph(hits, prescription)
```

## Edge truth labeling

`graphs.build` never looks at `particle_id` -- building the candidate graph
should work the same way it would on real data, where ground truth doesn't
exist. `graphs.truth` is a deliberately separate, optional second step that
*does* use it, for simulated data:

```python
from graphs.truth import label_edges, label_graph, purity

labeled_edges = label_edges(graph.nodes, graph.edges)   # or: label_graph(graph).edges
purity(labeled_edges)   # fraction of edges connecting the same particle
```

This adds an `is_true_edge` bool column (`graphs.edm.LABELED_EDGES_COLUMNS`):
`True` iff the edge's two hits share a `particle_id`. That's the standard
"same-track" truth definition used to train/score a GNN edge classifier --
note it does *not* also require the two hits to be adjacent along the
track, so a same-particle edge that skips a layer (e.g. a `ConnectionRules`
`delta_layer_id` range wider than 1) is still labeled true. `purity` raises
if `edges` hasn't been labeled yet, and is `nan` for an empty graph.

Both CLI subcommands take `--label-truth` to opt into this: `build` adds the
column to the written edges table and prints the graph's purity;
`visualize` labels on the fly and colors true edges more boldly (see
`graphs.vis.plot_edges_on`) so mislabeled/spurious connections stand out at
a glance.

## Setup

```bash
cd tracking/graphs
python3 -m venv .venv
.venv/bin/pip install -e ../../detector/detector2d -e ../../simulator/detectorsim2d -e ../../viz/style -e . -r requirements.txt
```

(`detector2d`/`detectorsim2d`/`viz_style` are sibling path packages, not on
PyPI, so they're installed explicitly rather than listed as a `graphs`
dependency.)

## Building a graph

```bash
.venv/bin/python -m graphs.cli build \
  --run-dir ../../simulator/detectorsim2d/out \
  --format arrow \
  --config configs/connection_rules.yaml \
  --output-dir out/
```

Reads `particles.<format>`/`hits.<format>` from `--run-dir` (a directory
already written by `detectorsim2d.cli run`), builds the graph under the given
prescription, and writes `out/edges.<format>`. With no `--config`, the
prescription defaults to `fully_connected`. Add `--label-truth` to include
the ground-truth `is_true_edge` column and print the graph's purity (see
"Edge truth labeling" below).

## Visualizing a graph

```bash
.venv/bin/python -m graphs.cli visualize \
  --run-dir ../../simulator/detectorsim2d/out --format arrow \
  --sim-config ../../simulator/detectorsim2d/configs/barrel6.yaml \
  --graph-config configs/connection_rules.yaml \
  --event-id 0 --save graph0.png
```

This is `detectorsim2d.vis.plot_event` (detector layers, trajectories, hits,
vertices -- see its own README) with the graph's edges for that event drawn
underneath the trajectories/hits as thin gray line segments between their
two hits' `(x, y)`. `--sim-config` is the *same* config the `detectorsim2d` run
used (for the detector layout to draw); `--graph-config` selects the
prescription; `--tracker-boundary` overrides the sim config's
`tracker_boundary` for this plot only, same as `detectorsim2d.cli visualize`.
Add `--label-truth` to color true edges more boldly (see "Edge truth
labeling" below).

## Using it as a library

```python
from detectorsim2d.io import read_run
from graphs.config import load_config
from graphs.build import build_graph
from graphs.vis import plot_event_with_graph

particles, hits = read_run("../../simulator/detectorsim2d/out", "arrow")
config = load_config("configs/connection_rules.yaml")
graph = build_graph(hits, config.prescription)

from detectorsim2d.config import load_config as load_sim_config
sim_config = load_sim_config("../../simulator/detectorsim2d/configs/barrel6.yaml")
fig = plot_event_with_graph(particles, hits, graph.edges, sim_config.layers, event_id=0)
```

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers prescription parsing (all three kinds, unknown-kind rejection);
config load/merge and the three example YAML files; edge construction for
each prescription (fully-connected undirected/directed, regional
sector-locality, connection-rule range gating including the phi-wraparound
case), edge-feature values, and that events never cross-connect; CSV/Arrow
IO round-trip; truth labeling and purity (including the unlabeled/empty
error cases); the visualization overlay (edge count, event filtering,
true/false edge coloring); and the `build`/`visualize` CLI subcommands end
to end, with and without `--label-truth`.
