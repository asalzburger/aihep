"""
Simple centroid (mean point) of all casualty dots - a 'single cluster' baseline.
Famously, John Snow's own analysis pointed to the Broad Street pump as the
outbreak's source; the centroid of all casualties should land close to it.
"""

import numpy as np
from svg_utils import load_dots, make_svg_to_latlon, render_cluster_svg

pts = load_dots()
pts_arr = np.array(pts)
svg_to_latlon = make_svg_to_latlon()

centroid = pts_arr.mean(axis=0)
lat, long_ = svg_to_latlon(centroid[0], centroid[1])
print(f"{len(pts)} points")
print(f"Centroid: x={centroid[0]:.1f}, y={centroid[1]:.1f}  ->  lat={lat:.6f}, long={long_:.6f}")

labels = [0] * len(pts)  # everything in one cluster

render_cluster_svg(
    out_path="out_centroid.svg",
    points=pts,
    labels=labels,
    centers=[tuple(centroid)],
    title="Centroid of all casualties (1 cluster)",
)
