# detectorsim2d

Particle event generation, CSV/Arrow IO, and visualization for a 2D particle
detector — tracker, electromagnetic and hadronic calorimeters, and a muon
system — built on top of [`detector/detector2d`](../../detector/detector2d)'s
geometry/intersection primitives. Mirrors the shape of
[`clustering/sensor`](../../clustering/sensor): an `edm.py`
schema module, a YAML-driven `config.py`, `simulate.py`, `io.py`, `vis.py`,
and a `cli.py` tying them together.

The physics content is one table: **the three interaction classes stop in
three different places**, which is the only reason a detector can tell them
apart at all.

| species | tracker | ECAL | HCAL | muon system |
|---|---|---|---|---|
| `electron`, `positron` | hits | **shower**, absorbed | — | — |
| `photon`, `pi0` | — (neutral) | **shower**, absorbed | — | — |
| `pi+`, `pi-` | hits | MIP trail | **shower**, absorbed | — |
| `neutron` | — (neutral) | — | **shower**, absorbed | — |
| `mu+`, `mu-` | hits | MIP trail | MIP trail | **hits** |

## Project layout

| module | contents |
|---|---|
| `detectorsim2d.edm` | The three per-event table schemas (`particles`, `hits`, `deposits`) shared by `simulate`, `io`, and `vis`. |
| `detectorsim2d.species` | The species registry: name, PDG code, charge, and interaction class (`em`/`hadron`/`muon`) -- the table above, as data. |
| `detectorsim2d.config` | `SimConfig`: detector layout (a flat list of `detector2d` layers, parsed by `detector2d.config`), `FieldConfig`/`FieldRegions`, `ParticleGunConfig`, `ResponseConfig`, loaded from YAML. The layout itself -- `layers:`/`detector:`/`calorimeter:`/`muon:` parsing -- is owned by [`detector2d.config`](../../detector/detector2d#describing-a-detector-layout-from-a-config-dict); see "Configuration" below. |
| `detectorsim2d.response` | The calorimeter response model: longitudinal layer fractions, a lateral Gaussian integrated exactly across each cell, MIP trails, and optional stochastic smearing -- see "Calorimeter response" below. |
| `detectorsim2d.simulate` | `propagate_particles(particles_df, layers, ...)` turns *any* particles table into `(hits, deposits)`; `hits_for_particles(...)` is its hits-only view, unchanged for tracker-only callers; `simulate_events(config)` additionally samples the particles from a `ParticleGunConfig`. |
| `detectorsim2d.io` | Read/write `particles`/`hits`/`deposits` tables as CSV or Apache Arrow. |
| `detectorsim2d.vis` | `plot_event(...)` (matplotlib x/y display), `plot_lego(...)` (the calorimeter unrolled in azimuth) and `export_svg(...)` (dependency-free raw SVG, exact circular arcs via native `A` path commands). |

`detectorsim2d.cli` wires `simulate` -> `io` -> `vis` into `run`/`visualize`
subcommands, the same split as `sensor.cli`.

## Setup

```bash
cd simulator/detectorsim2d
python3 -m venv .venv
.venv/bin/pip install -e ../../detector/detector2d -e ../../viz/style -e . -r requirements.txt
```

(`detector2d` and `viz_style` are sibling path packages, not on PyPI, so
they're installed explicitly rather than listed as a `detectorsim2d`
dependency.)

## Run the simulation

```bash
.venv/bin/python -m detectorsim2d.cli run \
  --config configs/full_detector.yaml \
  --n-events 100 \
  --output-dir out/ \
  --format arrow \
  --seed 42
```

Writes `out/particles.arrow`, `out/hits.arrow` and — if the layout has a
calorimeter — `out/deposits.arrow` (`--format csv` for `.csv`). All flags are
optional; with no `--config`, the layout defaults to an empty layer list
(particles, but no hits).

`configs/full_detector.yaml` is the whole detector (tracker + ECAL + HCAL +
muon system, piecewise field); `configs/barrel6.yaml` and
`configs/default.yaml` are the tracker-only layouts, unchanged.

## Visualize an event

```bash
.venv/bin/python -m detectorsim2d.cli visualize \
  --config configs/full_detector.yaml --output-dir out/ --format arrow \
  --event-id 0 --save event0.png
```

Drop `--save` for an interactive window. Each particle gets one color from a
fixed colorblind-safe palette; hits are outlined markers, vertices are stars,
calorimeter deposits are cell-shaped wedges shaded (and faded) by energy.
Tracker/muon sensors are solid lines, an idealized bare `CircleLayer` surface
is dashed, and a calorimeter ring is drawn as the radial slab it occupies.
With a piecewise `field:` in the config, each track is drawn along the path it
actually followed — bending one way in the tracker, straight through the
calorimeters, bending the other way in the muon system.

`--track-length` sets how far to draw a particle that has neither hits nor
deposits (default 100). `--tracker-boundary` overrides the config's
`tracker_boundary` (see "Configuration" below) for this plot only.

For the calorimeter unrolled in azimuth — one row per sampling layer, which
shows the longitudinal profile and lateral spread at a glance:

```bash
.venv/bin/python -m detectorsim2d.cli visualize \
  --config configs/full_detector.yaml --output-dir out/ --format arrow \
  --event-id 0 --view lego --save lego0.png
```

Add `--phi-range 92 108` to zoom in on one shower. Zoomed in far enough,
individual cell edges are drawn, which is the only way to actually *see* the
ECAL's half-cell stagger (at full scale a 256-cell ring's half cell is well
under a degree).

For a raw-SVG export instead (e.g. to overlay against a reference figure in
the *same* coordinate system, as `tracking/denby` does):

```python
from detectorsim2d.vis import export_svg

export_svg(config.layers, particles, hits, "event0.svg", width=900, height=900)
```

## Configuration

```yaml
layers:
  - {kind: line, layer_id: 0, p1: [10.0, -50.0], p2: [10.0, 50.0], pitch: 1.0}
  - {kind: circle, layer_id: 5, center: [0.0, 0.0], radius: 5.0, pitch: 0.5}

field:
  bz: 1.0          # constant field out of the 2D plane
  k: 0.2998        # R[len] = pt / (k * |q| * bz)

gun:
  n_particles: 3
  vertex_x: 0.0
  vertex_y: 0.0
  vertex_spread_x: 0.0
  vertex_spread_y: 2.0
  phi_min: -0.3
  phi_max: 0.3
  species: [electron, photon, "pi+", neutron, "mu+"]   # charge follows from species
  species_weights: []      # optional, one per species; empty = uniform
  charges: [-1.0, 1.0]     # legacy: used only when `species` is empty
  pt_min: 2.0
  pt_max: 10.0

n_events: 1
seed: null
tracker_boundary: null   # e.g. 210.0 -- outer tracker radius, see below
world_radius: null       # e.g. 800.0 -- outer edge of the whole detector
max_path_length: null    # e.g. 4000.0 -- hard cap; stops a curler looping forever
```

`layers`/`detector` — and the optional `calorimeter:`/`muon:` blocks — are
parsed entirely by
[`detector2d.config`](../../detector/detector2d#describing-a-detector-layout-from-a-config-dict)
(`build_layers_from_raw`, given the whole config dict) -- this package treats
the result as an opaque flat list of `detector2d` layer objects and doesn't
care which form built it. See `configs/barrel6.yaml` for a full working
`detector:` example (a 6-layer barrel detector with 3 inner "precision" layers
and 3 outer layers, `mode: detailed`), `configs/full_detector.yaml` for the
complete detector, and `detector2d`'s README for the layout-description
details.

### The full detector

`configs/full_detector.yaml`, in arbitrary length units continuing
`barrel6.yaml`'s scale:

| system | geometry | segmentation | field |
|---|---|---|---|
| tracker | 6 barrel layers, r = 29…200 | as `barrel6.yaml` | `Bz = +2.0` (strong) |
| ECAL | 3 rings, r = 225 / 255 / 285 | `n_phi = 256`; **layer 1 offset by half a cell** | `Bz = 0` |
| HCAL | 2 rings, r = 340 / 420 | `n_phi = 64` (coarser) | `Bz = 0` |
| muon | 3 octagonal stations, apothem 520 / 620 / 720 | each side a **triplet**: 3 planes 8 units apart | `Bz = -1.0` (half, reversed) |

The field is strong inside the solenoid, none in the calorimeters, and half
strength *reversed* in the muon system where the flux returns — so a track
curls one way in the tracker and the other way out in the muon chambers:

```yaml
field:
  k: 0.2998
  regions:
    - {r_max: 210.0, bz:  2.0}
    - {r_max: 480.0, bz:  0.0}
    - {bz: -1.0}
```

`world_radius` stops propagation once a particle has left the detector, and
`max_path_length` caps it outright — without that, a particle whose bend
radius is too small to escape the tracker circles forever. The shipped gun's
`pt_min: 20` is deliberately low enough that some particles do exactly that
and never reach the calorimeter at all; it's a real effect, and one
downstream code has to cope with.

## Calorimeter response

Two ingredients, both in `detectorsim2d/response.py` and both configurable under
`response:`:

**Longitudinally**, a shower's energy is split across layers by fixed
`layer_fractions` — largest first, then decreasing (ECAL `[0.60, 0.28, 0.12]`,
HCAL `[0.65, 0.35]`). Real showers build to a maximum a few radiation lengths
in and then decay; with only 3 (ECAL) / 2 (HCAL) sampling layers, a
monotonically decreasing profile is the honest simplification.

**Laterally**, energy is spread as a Gaussian in azimuth about the particle's
own impact point (taken from the *real* intersection, so a track that bent on
its way in deposits where it actually arrived), widening with depth. Each
cell's share is the **exact integral** of that Gaussian across the cell's
angular edges — an erf difference, not the density sampled at the cell center.
That matters specifically because the ECAL's middle layer is staggered by half
a cell: a shower centered on a boundary in layers 0 and 2 is centered mid-cell
in layer 1, and only an integral gets both right.

`mip_energy` is the *absolute* energy a minimum-ionizing particle leaves per
sampling layer — a fixed toll that does **not** scale with the particle's
energy, which is exactly what makes a muon's flat MIP trail distinguishable
from a shower. `stochastic` is the resolution term `sigma_E/E = a/sqrt(E)`;
it defaults to 0 in code (so results are exactly reproducible) and is turned
on in the shipped config.

```yaml
response:
  em:     {layer_fractions: [0.60, 0.28, 0.12], sigma_cells: 1.5, sigma_growth: 0.5}
  hadron: {layer_fractions: [0.65, 0.35],       sigma_cells: 1.0, sigma_growth: 0.4}
  mip_energy: {ecal: 0.3, hcal: 0.8}
  stochastic: {ecal: 0.10, hcal: 0.50}
```

`sigma_cells` is in units of the *layer's own* cells, so the ECAL's 1.5 fine
cells and the HCAL's 1.0 coarse cells still come out ~4x apart in absolute
width — a hadronic shower really is much broader than an EM one, which is why
the HCAL is segmented more coarsely in the first place.

### How a particle stops

Absorption is not a separate mechanism: each particle gets a **stopping
radius** from its species — the ECAL's outer edge for an EM particle, the
HCAL's for a hadron, `world_radius` for a muon — fed into the same cutoff
`tracker_boundary` already implemented. That single rule is why only muons
reach the muon system. The radii are derived from the calorimeter rings
themselves, not restated in config.

### Neutral particles leave no position hits

A tracking or muon chamber measures ionization, which a neutral particle does
not produce: a photon crosses the silicon invisibly and is seen for the first
time when it showers. **This is a behaviour change** from the pre-calorimeter
version, which gave a hit to anything that geometrically crossed a layer.
Particles with no `species` at all — a hand-built kinematic table, as
[`tracking/denby`](../../tracking/denby) produces — are exempt and behave
exactly as before: charged stubs that bend and leave hits, with no calorimeter
response.

Each event draws one nominal direction/charge/`pt` per particle from `gun`,
resolves a signed bend radius via
`detector2d.field.signed_radius(pt, charge, field.bz, field.k)` (`bz=0` or
`charge=0` -> a straight track), and keeps the earliest crossing of every
layer as that particle's hit.

If you already know a particle's `(x0, y0, phi0, charge, radius)` — e.g.
fit straight out of a reference picture, as in `tracking/denby` — skip the
gun entirely and call `hits_for_particles(your_particles_df, layers)`
directly.

**Multiple hits on one `layer_id` when the layout came from `detector:`
`mode: detailed` can come from two unrelated things**: the intended
module overlap (two hits close together in position and `path_length`), or
— for any curved (non-straight) trajectory starting exactly at the vertex —
a second crossing as its own circular arc loops back through the layer (far
apart in `path_length`). `mode: simplified` only ever reports the nearer of
those two; decomposing into modules surfaces both. See `detector2d`'s README
("Building a barrel layer") for why. If you want to isolate genuine overlap
hits, group `hits` by `(event_id, particle_id, layer_id)` and check that the
hits' `path_length` values are close together, not just that there's more
than one.

### `tracker_boundary` — no curling tracks

The constant field (`field.bz`) that bends a trajectory is only physical
inside the tracker volume; the idealized model has no concept of "leaving"
it, so a low-`pt` particle just keeps circling forever, re-crossing layers
it already left on the way back in (the "loop-back" crossing above). Setting

```yaml
tracker_boundary: 210.0   # e.g. just past barrel6.yaml's outermost radius (200)
```

makes `hits_for_particles`/`simulate_events` stop propagating each particle
the moment it first exits that radius (centered at the origin) —
`detector2d.intersect.first_intersection` against a bare `CircleLayer` at
that radius, via `detectorsim2d.simulate.boundary_crossing_s`. Any crossing
found beyond that point is dropped, not just hidden: it never reaches the
`hits` table. `detectorsim2d.vis` applies the same cutoff to the drawn curve of
a particle with no hits (which would otherwise be drawn out to
`--track-length`/`default_track_length` regardless of the boundary). Leave
it unset (`null`, the default) to keep the old unbounded behavior.

## Output schema (event data model)

Defined once in `detectorsim2d/edm.py`, reused by `simulate`, `io`, `vis`.
Joined by `event_id` (`hits`/`deposits` also by `particle_id`). The
hits/deposits split is the detector's own: a tracking layer answers *where*,
a calorimeter cell answers *how much*.

**`particles`**: `event_id, particle_id, species, pdg, x0, y0, phi0, charge,
energy, radius`

`species`/`pdg` may be NaN for a hand-built kinematic table. In this 2D toy
everything is transverse, so `energy` doubles as `pt` and is what bends.
`radius` is NaN/infinite for a straight track, and is the radius in the
*innermost* field region — with a piecewise field the particle's actual radius
changes region by region.

**`hits`**: `event_id, particle_id, system, layer_id, hit_id, x, y, s_local,
path_length`

`system` is `"tracker"` or `"muon"`, taken straight off the layer. `s_local`
is the hit's position along its layer; `path_length` is the arc length along
the particle's own trajectory to reach it.

**`deposits`**: `event_id, particle_id, system, layer_id, cell_id, x, y,
s_local, energy`

One row per (particle, cell) — i.e. **truth level**: two particles showering
into the same cell give two rows, which is what makes the table usable as
ground truth for cluster-splitting exercises. `detectorsim2d.response.sum_cells`
collapses it to the particle-blind view a real reconstruction would see.

`io.write_run` takes `deposits` as an optional fourth table and
`io.read_deposits` reads it back; **`io.read_run` deliberately stays a
`(particles, hits)` two-tuple**, because
[`tracking/graphs`](../../tracking/graphs) unpacks exactly two values from it.

## Using it as a library

```python
from detectorsim2d.config import SimConfig, ParticleGunConfig, FieldConfig
from detector2d.geometry import LineLayer

config = SimConfig(
    layers=[LineLayer(layer_id=0, p1=(10.0, -50.0), p2=(10.0, 50.0))],
    magnetic_field=FieldConfig(bz=1.0),
    gun=ParticleGunConfig(n_particles=5),
    n_events=20,
)
from detectorsim2d.simulate import simulate_events
particles, hits, deposits = simulate_events(config)
```

For the full detector, load the shipped config instead of hand-building one:

```python
from detectorsim2d.config import load_config
from detectorsim2d.simulate import simulate_events

config = load_config("configs/full_detector.yaml")
particles, hits, deposits = simulate_events(config)

# what each particle actually left behind
for _, particle in particles.iterrows():
    mine = deposits[deposits["particle_id"] == particle["particle_id"]]
    print(particle["species"], mine.groupby("system")["energy"].sum().to_dict())
```

## Tests

```bash
.venv/bin/python -m pytest tests/
```

Covers config load/merge, straight- and curved-particle hit geometry
(including the NaN-radius = straight-track path), seeded reproducibility,
CSV/Arrow round-trip, and the SVG exporter (straight vs. chunked-arc `d`
paths, element counts).

`test_response.py` is organized as **one case per particle type**, pinning
each species' signature from the table at the top of this README: that an EM
particle's shower is fully contained in the ECAL with decreasing per-layer
energies and never reaches the muon system; that a photon leaves zero tracker
hits but a full shower; that a charged hadron's ECAL trail is one MIP cell per
layer and does not scale with its energy while its HCAL shower does; that a
neutron leaves nothing before the HCAL; that a muon gets exactly 9 muon hits
and reverses its curvature in the flux return; and that a shower aimed at a
cell boundary splits *exactly* evenly in the unstaggered ECAL layers while the
staggered one resolves it.
