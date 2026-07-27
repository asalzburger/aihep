"""
Extract red-dot (cholera casualty) coordinates from the John Snow map SVG overlay.

Input:  overlay.svg  (Soho_map_ticks_overlay_cleaned.svg) - contains ~500+ small
        circular <path> elements filled with #e8000d marking each casualty.
Output: dots.csv - one row per dot, with its center (x, y) in SVG user units
        (viewBox coordinate space, y grows downward).
"""

import re
import csv

SVG_PATH = "overlay.svg"
OUT_CSV = "dots.csv"
RED_FILL = "#e8000d"


def extract_dot_centers(svg_path: str, fill: str = RED_FILL):
    content = open(svg_path, "r", encoding="utf-8").read()

    # Match <path ... style="...fill:#e8000d..." ... d="..."/>
    pattern = re.compile(
        r'<path[^>]*style="[^"]*fill:' + re.escape(fill) + r'[^"]*"[^>]*d="([^"]+)"'
    )
    paths = pattern.findall(content)

    centers = []
    for d in paths:
        nums = [float(n) for n in re.findall(r"-?\d+\.?\d*", d)]
        xs = nums[0::2]
        ys = nums[1::2]
        cx = (min(xs) + max(xs)) / 2
        cy = (min(ys) + max(ys)) / 2
        centers.append((cx, cy))

    return centers


if __name__ == "__main__":
    centers = extract_dot_centers(SVG_PATH)
    print(f"Extracted {len(centers)} red dots (casualty markers).")

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])
        writer.writerows(centers)

    print(f"Saved coordinates to {OUT_CSV}")
