"""
DBSCAN clustering on casualty dots under a few similar (eps, min_samples)
configurations, chosen from the nearest-neighbor distance distribution of the
data (mean 1st-NN distance ~30 units, mean 4th-NN distance ~87 units, in SVG
coordinate units on a 4417x4201 canvas).

Cluster centers are reported as the mean position of each cluster's points;
noise points (label -1) are kept in the SVG in gray but excluded from center
computation.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from svg_utils import load_dots, render_cluster_svg

pts = load_dots()
pts_arr = np.array(pts)

CONFIGS = [
    dict(eps=50, min_samples=4),
    dict(eps=80, min_samples=5),
    dict(eps=120, min_samples=5),
]

for cfg in CONFIGS:
    db = DBSCAN(**cfg)
    labels = db.fit_predict(pts_arr)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())

    centers = []
    for cid in sorted(set(labels)):
        if cid == -1:
            continue
        mask = labels == cid
        centers.append(tuple(pts_arr[mask].mean(axis=0)))

    print(f"\n--- DBSCAN eps={cfg['eps']}, min_samples={cfg['min_samples']} ---")
    print(f"clusters={n_clusters}, noise points={n_noise}")
    for i, c in enumerate(centers):
        n = int((labels == i).sum())
        print(f"cluster {i}: center=({c[0]:.1f}, {c[1]:.1f}), n={n}")

    tag = f"eps{cfg['eps']}_ms{cfg['min_samples']}"
    render_cluster_svg(
        out_path=f"out_dbscan_{tag}.svg",
        points=pts,
        labels=labels.tolist(),
        centers=centers,
        title=f"DBSCAN (eps={cfg['eps']}, min_samples={cfg['min_samples']}) - {n_clusters} clusters",
    )
