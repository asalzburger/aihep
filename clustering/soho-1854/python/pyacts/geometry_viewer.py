#!/usr/bin/env python3
# This file is part of the ACTS project
#
# Copyright (C) 2016 CERN for the benefit of the ACTS project
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

import argparse
import math
import os
from collections import defaultdict

os.environ["ACTS_SEQUENCER_DISABLE_FPEMON"] = "1"

import acts
import acts.examples
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button, CheckButtons


def _geometry_id(surf):
    gid = getattr(surf, "geometryId", None)
    return gid() if callable(gid) else gid


def _is_sensitive(surf):
    gid = _geometry_id(surf)
    if gid is None:
        return False
    sensitive = getattr(gid, "sensitive", 0)
    return sensitive() > 0 if callable(sensitive) else sensitive > 0


def _surface_type_name(surf):
    stype = getattr(surf, "type", None)
    stype = stype() if callable(stype) else stype
    return str(stype)


def _surface_center_xyz(surf, geo_context):
    c = surf.center(geo_context)
    # acts::Vector3 python wrapper is indexable
    return float(c[0]), float(c[1]), float(c[2])


def _iter_poly_vertices(poly):
    vtx = getattr(poly, "vertices", None)
    if vtx is None:
        return []
    values = vtx() if callable(vtx) else vtx
    out = []
    for v in values:
        out.append((float(v[0]), float(v[1]), float(v[2])))
    return out


def _iter_poly_faces(poly, n_vertices):
    for aname in ("faces", "triangles", "mesh"):
        face_attr = getattr(poly, aname, None)
        if face_attr is None:
            continue
        values = face_attr() if callable(face_attr) else face_attr
        out = []
        for face in values:
            try:
                idx = [int(i) for i in face]
            except TypeError:
                continue
            if len(idx) >= 3 and all(0 <= i < n_vertices for i in idx):
                out.append(idx)
        if out:
            return out
    return []


def _surface_polygons_xyz(surf, geo_context, view_config):
    # Binding signatures vary across ACTS versions; try known variants.
    for mname in ("polyhedronRepresentation", "polyhedronRepresenation"):
        method = getattr(surf, mname, None)
        if method is None:
            continue
        for call_args in (
            (geo_context, view_config),
            (geo_context,),
            (view_config,),
            tuple(),
        ):
            try:
                poly = method(*call_args)
                verts = _iter_poly_vertices(poly)
                if not verts:
                    continue
                faces = _iter_poly_faces(poly, len(verts))
                if faces:
                    return [[verts[i] for i in f] for f in faces]
                # Fallback: polygon without explicit connectivity
                return [verts]
            except TypeError:
                continue
            except Exception:
                break
    return [[]]


def _collect_sensitive_surfaces(detector, tracking_geometry, geo_context):
    surfaces = []
    if hasattr(detector, "volumePtrs"):
        for vol in detector.volumePtrs():
            if not hasattr(vol, "surfacePtrs"):
                continue
            for surf in vol.surfacePtrs():
                if _is_sensitive(surf):
                    surfaces.append(surf)
        if surfaces:
            return surfaces

    if hasattr(tracking_geometry, "visitSurfaces"):
        def _visitor(surf):
            if _is_sensitive(surf):
                surfaces.append(surf)
        tracking_geometry.visitSurfaces(_visitor)
        if surfaces:
            return surfaces

    raise RuntimeError(
        "Could not extract sensitive surfaces from detector/tracking geometry with current bindings."
    )


def _build_detector(detector_name):
    if detector_name == "odd":
        from acts.examples.odd import getOpenDataDetector

        detector = getOpenDataDetector(None)
        return detector, detector.trackingGeometry()

    detector = acts.examples.GenericDetector(acts.examples.GenericDetector.Config())
    return detector, detector.trackingGeometry()


def _order_polygon_points(points_2d):
    if len(points_2d) < 3:
        return points_2d
    cx = sum(p[0] for p in points_2d) / len(points_2d)
    cy = sum(p[1] for p in points_2d) / len(points_2d)
    return sorted(points_2d, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))


def _projection_key(points_2d, eps):
    if not points_2d:
        return None
    cx = sum(p[0] for p in points_2d) / len(points_2d)
    cy = sum(p[1] for p in points_2d) / len(points_2d)
    return (round(cx / eps), round(cy / eps))


def _make_interactive_plot(surface_shapes):
    types = sorted({s["type"] for s in surface_shapes})
    enabled = {t: True for t in types}
    grouped = defaultdict(list)
    for shape in surface_shapes:
        grouped[shape["type"]].append(shape)

    fig, (ax_xy, ax_rz) = plt.subplots(1, 2, figsize=(14, 7))
    plt.subplots_adjust(left=0.24, bottom=0.12)

    # Futuristic neon theme
    bg = "#080b16"
    panel = "#10162a"
    text = "#d8ecff"
    grid = "#2de2e6"
    neon_palette = [
        "#00f5d4",
        "#00bbf9",
        "#f15bb5",
        "#9b5de5",
        "#fee440",
        "#00f0ff",
        "#ff6ad5",
        "#7df9ff",
    ]
    type_color = {t: neon_palette[i % len(neon_palette)] for i, t in enumerate(types)}

    fig.patch.set_facecolor(bg)
    ax_xy.set_facecolor(panel)
    ax_rz.set_facecolor(panel)

    ax_xy.set_title("Sensitive surfaces: x-y")
    ax_xy.set_xlabel("x [mm]")
    ax_xy.set_ylabel("y [mm]")
    ax_xy.set_aspect("equal", adjustable="box")
    ax_xy.grid(True, color=grid, alpha=0.18, linewidth=0.5)

    ax_rz.set_title("Sensitive surfaces: r-z")
    ax_rz.set_xlabel("z [mm]")
    ax_rz.set_ylabel("r [mm]")
    ax_rz.grid(True, color=grid, alpha=0.18, linewidth=0.5)

    for ax in (ax_xy, ax_rz):
        ax.title.set_color(text)
        ax.xaxis.label.set_color(text)
        ax.yaxis.label.set_color(text)
        ax.tick_params(colors=text)
        for spine in ax.spines.values():
            spine.set_color("#2a3d63")

    artists_xy = defaultdict(list)
    artists_rz = defaultdict(list)
    seen_xy = set()
    seen_rz = set()
    label_added = {t: False for t in types}
    dedup_eps = 0.5  # mm in projected coordinates

    n_input = len(surface_shapes)
    n_xy_drawn = 0
    n_rz_drawn = 0
    n_xy_suppressed = 0  # valid bin but already drawn in x-y
    n_rz_suppressed = 0
    n_no_projection = 0  # no centroid for either view
    n_skip_both = 0  # nothing drawn in either view

    for tname in types:
        chunk = grouped[tname]
        color = type_color[tname]
        for shape in chunk:
            polygons = shape["polygons"]
            all_verts = [v for poly in polygons for v in poly]
            xy_points = _order_polygon_points([(v[0], v[1]) for v in all_verts])
            rz_points = _order_polygon_points(
                [(v[2], math.hypot(v[0], v[1])) for v in all_verts]
            )
            key_xy = _projection_key(xy_points, dedup_eps)
            key_rz = _projection_key(rz_points, dedup_eps)

            if key_xy is None and key_rz is None:
                n_no_projection += 1

            if key_xy is not None and key_xy in seen_xy:
                n_xy_suppressed += 1
            if key_rz is not None and key_rz in seen_rz:
                n_rz_suppressed += 1

            draw_xy = key_xy is not None and key_xy not in seen_xy
            draw_rz = key_rz is not None and key_rz not in seen_rz
            if not draw_xy and not draw_rz:
                n_skip_both += 1
                continue

            label_pending = tname if not label_added[tname] and draw_xy else None

            for poly in polygons:
                xy_poly = _order_polygon_points([(v[0], v[1]) for v in poly])
                rz_poly = _order_polygon_points(
                    [(v[2], math.hypot(v[0], v[1])) for v in poly]
                )

                if draw_xy and len(xy_poly) >= 3:
                    p_xy = Polygon(
                        xy_poly,
                        closed=True,
                        fill=True,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=1.0,
                        alpha=0.14,
                        label=label_pending,
                    )
                    ax_xy.add_patch(p_xy)
                    artists_xy[tname].append(p_xy)
                    label_pending = None
                elif draw_xy and len(xy_poly) == 2:
                    line_xy, = ax_xy.plot(
                        [xy_poly[0][0], xy_poly[1][0]],
                        [xy_poly[0][1], xy_poly[1][1]],
                        color=color,
                        linewidth=1.0,
                        alpha=0.8,
                        label=label_pending,
                    )
                    artists_xy[tname].append(line_xy)
                    label_pending = None
                elif draw_xy and len(xy_poly) == 1:
                    sc_xy = ax_xy.scatter(
                        [xy_poly[0][0]],
                        [xy_poly[0][1]],
                        c=color,
                        s=12,
                        alpha=0.9,
                        label=label_pending,
                    )
                    artists_xy[tname].append(sc_xy)
                    label_pending = None

                if draw_rz and len(rz_poly) >= 3:
                    p_rz = Polygon(
                        rz_poly,
                        closed=True,
                        fill=True,
                        facecolor=color,
                        edgecolor=color,
                        linewidth=1.0,
                        alpha=0.14,
                    )
                    ax_rz.add_patch(p_rz)
                    artists_rz[tname].append(p_rz)
                elif draw_rz and len(rz_poly) == 2:
                    line_rz, = ax_rz.plot(
                        [rz_poly[0][0], rz_poly[1][0]],
                        [rz_poly[0][1], rz_poly[1][1]],
                        color=color,
                        linewidth=1.0,
                        alpha=0.8,
                    )
                    artists_rz[tname].append(line_rz)
                elif draw_rz and len(rz_poly) == 1:
                    sc_rz = ax_rz.scatter(
                        [rz_poly[0][0]], [rz_poly[0][1]], c=color, s=12, alpha=0.9
                    )
                    artists_rz[tname].append(sc_rz)

            if draw_xy:
                seen_xy.add(key_xy)
                label_added[tname] = True
                n_xy_drawn += 1
            if draw_rz:
                seen_rz.add(key_rz)
                n_rz_drawn += 1

    n_art_xy = sum(len(a) for a in artists_xy.values())
    n_art_rz = sum(len(a) for a in artists_rz.values())
    pct_xy = (100.0 * n_xy_suppressed / n_input) if n_input else 0.0
    pct_rz = (100.0 * n_rz_suppressed / n_input) if n_input else 0.0
    pct_skip = (100.0 * n_skip_both / n_input) if n_input else 0.0

    print(
        "[geometry_viewer] Deduplication (centroid bin eps={} mm in projected coordinates):".format(
            dedup_eps
        )
    )
    print("  Input sensitive surfaces: {}".format(n_input))
    print(
        "  x-y: drew {:,} unique bins; {:,} surfaces suppressed (duplicate bin) ({:.1f}% of input)".format(
            n_xy_drawn, n_xy_suppressed, pct_xy
        )
    )
    print(
        "  r-z: drew {:,} unique bins; {:,} surfaces suppressed (duplicate bin) ({:.1f}% of input)".format(
            n_rz_drawn, n_rz_suppressed, pct_rz
        )
    )
    print(
        "  Skipped in both views (nothing new to draw): {:,} ({:.1f}% of input)".format(
            n_skip_both, pct_skip
        )
    )
    print("  No projection centroid (empty / degenerate): {:,}".format(n_no_projection))
    print(
        "  Matplotlib artists: x-y={:,}, r-z={:,}".format(n_art_xy, n_art_rz)
    )

    ax_xy.relim()
    ax_xy.autoscale_view()
    ax_rz.relim()
    ax_rz.autoscale_view()

    ax_legend = plt.axes([0.02, 0.24, 0.2, 0.7])
    ax_legend.set_facecolor(panel)
    for spine in ax_legend.spines.values():
        spine.set_color("#2a3d63")
    ax_legend.set_title("Surface types")
    checks = CheckButtons(ax_legend, types, [True] * len(types))
    ax_legend.title.set_color(text)
    if hasattr(checks, "labels"):
        for lbl, t in zip(checks.labels, types):
            lbl.set_color(type_color[t])
    if hasattr(checks, "rectangles"):
        for rect in checks.rectangles:
            rect.set_facecolor("#0b1325")
            rect.set_edgecolor("#35507b")

    xy_home = (ax_xy.get_xlim(), ax_xy.get_ylim())
    rz_home = (ax_rz.get_xlim(), ax_rz.get_ylim())

    def _toggle(label):
        enabled[label] = not enabled[label]
        for art in artists_xy[label]:
            art.set_visible(enabled[label])
        for art in artists_rz[label]:
            art.set_visible(enabled[label])
        fig.canvas.draw_idle()

    checks.on_clicked(_toggle)

    ax_reset = plt.axes([0.02, 0.12, 0.08, 0.06])
    ax_reset.set_facecolor(panel)
    reset_btn = Button(ax_reset, "Reset")
    reset_btn.label.set_color(text)
    reset_btn.color = "#12203d"
    reset_btn.hovercolor = "#1f3464"

    def _reset(_event):
        ax_xy.set_xlim(*xy_home[0])
        ax_xy.set_ylim(*xy_home[1])
        ax_rz.set_xlim(*rz_home[0])
        ax_rz.set_ylim(*rz_home[1])
        fig.canvas.draw_idle()

    reset_btn.on_clicked(_reset)
    leg = ax_xy.legend(loc="upper right", fontsize=8, facecolor=panel, edgecolor="#2a3d63")
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_color(text)
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive ACTS sensitive-surface viewer in x-y and r-z"
    )
    parser.add_argument(
        "--detector",
        choices=("generic", "odd"),
        default="generic",
        help="Detector model to visualize",
    )
    args = parser.parse_args()

    detector, tracking_geometry = _build_detector(args.detector)
    geo_context = acts.GeometryContext()
    surfaces = _collect_sensitive_surfaces(detector, tracking_geometry, geo_context)

    view_config = acts.ViewConfig()

    surface_shapes = []
    for surf in surfaces:
        polygons = _surface_polygons_xyz(surf, geo_context, view_config)
        if not polygons or not polygons[0]:
            # Fallback to center point if polyhedron export is unavailable.
            polygons = [[_surface_center_xyz(surf, geo_context)]]
        surface_shapes.append({"polygons": polygons, "type": _surface_type_name(surf)})

    if not surface_shapes:
        raise RuntimeError("No sensitive surfaces found.")

    print(f"Loaded {len(surface_shapes)} sensitive surfaces")
    _make_interactive_plot(surface_shapes)


if __name__ == "__main__":
    main()
