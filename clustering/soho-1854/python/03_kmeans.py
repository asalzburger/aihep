"""
K-means clustering on casualty dots, for k = 1, 3, 5.
Produces one SVG + prints cluster centers/sizes per k.
"""

import numpy as np
from sklearn.cluster import KMeans
from svg_utils import load_dots, render_cluster_svg

pts = load_dots()
pts_arr = np.array(pts)

for k in (1, 3, 5):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(pts_arr)
    centers = km.cluster_centers_

    print(f"\n--- k-means, k={k} ---")
    for i, c in enumerate(centers):
        n = int((labels == i).sum())
        print(f"cluster {i}: center=({c[0]:.1f}, {c[1]:.1f}), n={n}")

    render_cluster_svg(
        out_path=f"out_kmeans_k{k}.svg",
        points=pts,
        labels=labels.tolist(),
        centers=[tuple(c) for c in centers],
        title=f"k-means (k={k})",
    )
