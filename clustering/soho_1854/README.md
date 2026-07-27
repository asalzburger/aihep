# soho_1854

Clustering exercise on John Snow's 1854 Soho cholera outbreak map: extract
the ~536 casualty markers from an annotated SVG of the map, cluster them a
few different ways (simple centroid, k-means, DBSCAN), convert pixel
coordinates to real lat/long, and render the results back onto the map as
SVG overlays — plus an animated build-up of the analysis.

Famously, John Snow's own manual analysis pointed to the Broad Street water
pump as the outbreak's source; the point of this exercise is to see how
close a few standard clustering methods land to that pump using nothing but
the casualty locations.

## Project layout

```
soho_1854/
  python/
    svg_utils.py         # shared helpers: load_dots, make_svg_to_latlon, render_cluster_svg
    01_extract_dots.py    # parse the annotated SVG -> resources/dots.csv
    02_centroid.py          # single-cluster baseline: mean of all casualties
    03_kmeans.py              # k-means for k = 1, 3, 5
    04_dbscan.py                # DBSCAN for a few (eps, min_samples) configs
    05_animation.py               # reveal/converge/zoom animation -> animation.mp4 + preview.gif
  resources/
    Soho_map_raw.svg            # plain line-art base map (used as the render background)
    Soho_map_annoted.svg         # base map + red casualty dot overlay (input to 01)
    Soho_map_annoted_w_pump.svg   # + the real Broad Street pump marked
    Soho_map_vector_gauge.svg      # reference map used to pick gauge.csv's control points
    dots.csv                        # extracted casualty coordinates (output of 01, committed)
    gauge.csv                        # 3 (x, y) -> (lat, long) control points for the affine fit
  tests/
```

The numbered scripts are plain, sequential scripts (not a package) — each
one is meant to be run directly with `python/` as the working directory,
since they use paths like `../resources/dots.csv` and import `svg_utils`
as a sibling module.

## Setup

```bash
cd clustering/soho_1854
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

`ffmpeg` must additionally be on `PATH` to actually encode
`animation.mp4`/`preview.gif` in `05_animation.py` (e.g. `brew install
ffmpeg`) — everything else only needs the pip requirements.

## Running the analysis

All scripts are run from `python/`, in order:

```bash
cd python
../.venv/bin/python 01_extract_dots.py   # regenerates ../resources/dots.csv
../.venv/bin/python 02_centroid.py       # -> out_centroid.svg
../.venv/bin/python 03_kmeans.py         # -> out_kmeans_k1.svg, out_kmeans_k3.svg, out_kmeans_k5.svg
../.venv/bin/python 04_dbscan.py         # -> out_dbscan_eps50_ms4.svg, out_dbscan_eps80_ms5.svg, out_dbscan_eps120_ms5.svg
../.venv/bin/python 05_animation.py      # -> animation.mp4, preview.gif (needs ffmpeg)
```

`01_extract_dots.py` regenerates `resources/dots.csv` from the annotated
SVG; the extraction is deterministic, so re-running it should reproduce the
committed file byte-for-byte. `02`–`04` all read `resources/dots.csv` and
render their cluster centers/assignments as an SVG overlay on
`resources/Soho_map_raw.svg`, printing each cluster's center in both pixel
and lat/long coordinates (via the affine fit in `svg_utils.make_svg_to_latlon`,
calibrated from `resources/gauge.csv`).

## Tests

```bash
.venv/bin/python -m pytest tests/
```

- **`test_svg_utils.py`** — unit tests for the shared helpers: `load_dots`,
  the affine `make_svg_to_latlon` fit (reproduced against a known transform,
  checked at a held-out point), `_base_map_paths` SVG extraction, and
  `render_cluster_svg`'s output (cluster colors, noise color, title).
- **`test_extract_dots.py`** — `extract_dot_centers` against a small
  synthetic SVG, and against the real `Soho_map_annoted.svg`, checked to
  match the committed `dots.csv` exactly.
- **`test_scripts.py`** — end-to-end smoke tests that actually run
  `01`–`04` as subprocesses (matching how they're documented to be used
  above) against a throwaway copy of the project, and check they exit
  cleanly and produce the expected output files. Never touches the real
  `resources/` directory.
- **`test_animation.py`** — unit tests for `05_animation.py`'s pure helpers
  (`ease_in_out`, base-map rendering). Skips cleanly if `cairosvg`/`pillow`
  aren't installed; doesn't invoke `main()` (frame-by-frame rendering +
  `ffmpeg` encoding is slow and produces large binary output, not a good
  fit for a fast test suite).
