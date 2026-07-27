"""
K-means clustering on casualty dots, for k = 1, 3, 5.
Produces one SVG + prints cluster centers/sizes per k.
"""

import numpy as np
from sklearn.cluster import KMeans
from svg_utils import load_dots, make_svg_to_latlon, render_cluster_svg

pts = load_dots()
pts_arr = np.array(pts)
svg_to_latlon = make_svg_to_latlon()

for k in (1, 3, 5):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(pts_arr)
    centers = km.cluster_centers_

    print(f"\n--- k-means, k={k} ---")
    for i, c in enumerate(centers):
        n = int((labels == i).sum())
        lat, long_ = svg_to_latlon(c[0], c[1])
        print(
            f"cluster {i}: center=({c[0]:.1f}, {c[1]:.1f}), n={n}  ->  "
            f"lat={lat:.6f}, long={long_:.6f}"
        )

    render_cluster_svg(
        out_path=f"out_kmeans_k{k}.svg",
        points=pts,
        labels=labels.tolist(),
        centers=[tuple(c) for c in centers],
        title=f"k-means (k={k})",
    )
