"""
Build an animation with three acts, rendered on top of the John Snow Soho
base map:

  1. Intro    - the bare map, no casualties marked.
  2. Reveal   - the 536 casualty dots appear in random-order batches; after
                each batch the running centroid is recomputed and drawn,
                together with a shaded uncertainty region (~95% CI on the
                mean, shrinking as 1/sqrt(n) with each new batch).
  3. Converge - the uncertainty region shrinks to nothing and the marker
                settles on the final centroid.
  4. Zoom     - the camera zooms in on the final centroid -> Broad Street.
  5. Pump     - the real Broad Street pump is marked in blue, right next to
                the computed centroid, for comparison.

Output: animation.mp4 (full res) and preview.gif (lightweight preview).

Approach: render the static line-art map once at a high working resolution
with cairosvg, then do all per-frame compositing (dots, marker, zoom crop)
with Pillow, which is far faster than re-rendering SVG per frame. ffmpeg
encodes the final PNG frame sequence to video.
"""

import io
import math
import os
import random
import shutil
import subprocess

import cairosvg
import numpy as np
from PIL import Image, ImageDraw

from svg_utils import load_dots, _base_map_paths, WIDTH, HEIGHT, BASE_MAP_PATH

FRAME_DIR = "/tmp/soho_animation_frames"
FPS = 24
HIRES_W = 3200  # working resolution used for compositing / zoom source
OUT_W = 1400  # final video resolution
RED = (232, 0, 13)
BLACK = (0, 0, 0)
GOLD = (30, 30, 30)
BLUE = (20, 90, 230)

# Real Broad Street pump location, hand-located on the base map (native
# SVG coordinate space, 4417x4201) by inspecting the "PUMP" glyph near
# Broad St / Cambridge St.
PUMP_NATIVE = (2483.0, 1918.0)

random.seed(42)


def build_clean_base_svg():
    """Base map, no dots, white background - same look as base_map.svg."""
    paths = _base_map_paths(BASE_MAP_PATH)
    return (
        f'<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg version="1.1" xmlns="http://www.w3.org/2000/svg" '
        f'width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
        f'<rect x="0" y="0" width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>'
        f'<g id="base_map">{paths}</g>'
        f"</svg>"
    )


def render_base_png(hires_w: int) -> Image.Image:
    svg_str = build_clean_base_svg()
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode(), output_width=hires_w)
    return Image.open(io.BytesIO(png_bytes)).convert("RGB")


def draw_dot(draw: ImageDraw.ImageDraw, x, y, r, color=RED):
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(0, 0, 0))


def draw_marker(draw: ImageDraw.ImageDraw, x, y, r, color=BLACK, width=5):
    if r <= 0:
        return
    draw.line([x - r, y, x + r, y], fill=color, width=width)
    draw.line([x, y - r, x, y + r], fill=color, width=width)
    draw.ellipse([x - r, y - r, x + r, y + r], outline=color, width=max(2, width - 2))


def draw_pump_marker(draw: ImageDraw.ImageDraw, x, y, r, color=BLUE, width=5):
    """Solid pin-style marker for the real Broad Street pump location."""
    if r <= 0:
        return
    draw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(255, 255, 255), width=width)
    inner = max(2, r * 0.28)
    draw.ellipse([x - inner, y - inner, x + inner, y + inner], fill=(255, 255, 255))


def draw_uncertainty(canvas_rgba: Image.Image, x, y, r, color=(20, 90, 230, 70)) -> Image.Image:
    """Alpha-blend a shaded confidence disc for the running centroid estimate."""
    if r <= 0.5:
        return canvas_rgba
    overlay = Image.new("RGBA", canvas_rgba.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.ellipse([x - r, y - r, x + r, y + r], fill=color, outline=(20, 90, 230, 180), width=3)
    return Image.alpha_composite(canvas_rgba, overlay)


def ease_in_out(t: float) -> float:
    return t * t * (3 - 2 * t)  # smoothstep


def main():
    if os.path.exists(FRAME_DIR):
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR)

    print("Rendering base map...")
    base_hires = render_base_png(HIRES_W)
    hw, hh = base_hires.size
    scale = hw / WIDTH
    print(f"Base map rendered at {hw}x{hh} (scale={scale:.4f})")

    dots = load_dots()
    dots_hires = [(x * scale, y * scale) for x, y in dots]
    n = len(dots_hires)

    centroid = (
        sum(x for x, _ in dots) / n,
        sum(y for _, y in dots) / n,
    )
    centroid_hires = (centroid[0] * scale, centroid[1] * scale)
    print(f"Centroid (native): ({centroid[0]:.1f}, {centroid[1]:.1f})")

    order = list(range(n))
    random.shuffle(order)

    # Points in reveal order, for running (batched) centroid + uncertainty.
    dots_ordered = np.array([dots_hires[i] for i in order])
    cumsum = np.cumsum(dots_ordered, axis=0)

    # Fixed spread of the full dataset, used as the population std in a
    # standard-error-of-the-mean style estimate: SE(n) = z * std / sqrt(n).
    # This shrinks smoothly and monotonically as batches accumulate, unlike
    # a running sample std (which is noisy/undefined for tiny n).
    std_x, std_y = dots_ordered[:, 0].std(), dots_ordered[:, 1].std()
    pooled_std = math.sqrt((std_x ** 2 + std_y ** 2) / 2)
    Z = 1.96  # ~95% CI

    def uncertainty_radius(count):
        return Z * pooled_std / math.sqrt(max(count, 1))

    dot_r_hires = 14 * scale  # native dot radius was 14 units
    small_marker_r = 16 * scale  # running-estimate marker size during reveal

    # ---- phase durations (seconds) ----
    T_INTRO = 1.5
    N_BATCHES = 24  # how many "clustering runs" during the reveal
    T_REVEAL = 3.6
    T_CONVERGE = 0.9
    T_HOLD = 0.8
    T_ZOOM = 2.5
    T_PUMP = 0.7
    T_ENDHOLD = 2.2

    def n_frames(t):
        return max(1, round(t * FPS))

    f_intro = n_frames(T_INTRO)
    f_reveal = n_frames(T_REVEAL)
    f_converge = n_frames(T_CONVERGE)
    f_hold = n_frames(T_HOLD)
    f_zoom = n_frames(T_ZOOM)
    f_pump = n_frames(T_PUMP)
    f_endhold = n_frames(T_ENDHOLD)
    frames_per_batch = max(1, f_reveal // N_BATCHES)

    frame_idx = 0

    def save_frame(img: Image.Image):
        nonlocal frame_idx
        img.save(os.path.join(FRAME_DIR, f"f_{frame_idx:05d}.png"))
        frame_idx += 1

    full_box = (0, 0, hw, hh)
    aspect = OUT_W / round(OUT_W * HEIGHT / WIDTH)
    out_h = round(OUT_W * HEIGHT / WIDTH)

    def crop_resize(img: Image.Image, box):
        return img.crop(box).resize((OUT_W, out_h), Image.LANCZOS)

    # ---- Act 1: intro, bare map ----
    print("Act 1: intro...")
    frame = crop_resize(base_hires, full_box)
    for _ in range(f_intro):
        save_frame(frame)

    # ---- Act 2: dots reveal in batches; centroid re-clustered per batch ----
    # Dots fade in smoothly and continuously (for a nice reveal), but the
    # centroid estimate + its uncertainty band only update once per batch,
    # like re-running the clustering on progressively more data.
    print("Act 2: batched reveal + running centroid...")
    max_r = 55 * scale  # final, fully-converged marker size

    batch_frame_counts = []
    acc = 0
    for b in range(N_BATCHES):
        fcount = (f_reveal - acc) if b == N_BATCHES - 1 else frames_per_batch
        batch_frame_counts.append(fcount)
        acc += fcount

    frame_in_reveal = 0
    r_unc_final = uncertainty_radius(n)
    for b, fcount in enumerate(batch_frame_counts):
        target_count = n if b == N_BATCHES - 1 else round((b + 1) / N_BATCHES * n)
        target_count = max(1, target_count)
        running_mean = cumsum[target_count - 1] / target_count
        r_unc = uncertainty_radius(target_count)

        for _ in range(fcount):
            frame_in_reveal += 1
            count = round(frame_in_reveal / f_reveal * n)
            canvas = base_hires.copy().convert("RGBA")
            draw = ImageDraw.Draw(canvas)
            for idx in order[:count]:
                x, y = dots_hires[idx]
                draw_dot(draw, x, y, dot_r_hires)
            canvas = draw_uncertainty(canvas, running_mean[0], running_mean[1], r_unc)
            draw = ImageDraw.Draw(canvas)
            draw_marker(draw, running_mean[0], running_mean[1], small_marker_r, width=4)
            save_frame(crop_resize(canvas.convert("RGB"), full_box))

    # clean map + all dots, no marker/shading - used as the base for later acts
    all_dots_img = base_hires.copy()
    draw = ImageDraw.Draw(all_dots_img)
    for x, y in dots_hires:
        draw_dot(draw, x, y, dot_r_hires)

    # ---- Act 3: converge - uncertainty shrinks to 0, marker settles ----
    print("Act 3: converge on final centroid...")
    for i in range(f_converge):
        t = ease_in_out((i + 1) / f_converge)
        r_unc = r_unc_final * (1 - t)
        r_marker = small_marker_r + (max_r - small_marker_r) * t
        canvas = all_dots_img.copy().convert("RGBA")
        canvas = draw_uncertainty(canvas, centroid_hires[0], centroid_hires[1], r_unc)
        draw = ImageDraw.Draw(canvas)
        draw_marker(draw, centroid_hires[0], centroid_hires[1], r_marker)
        save_frame(crop_resize(canvas.convert("RGB"), full_box))

    final_with_marker = all_dots_img.copy()
    draw = ImageDraw.Draw(final_with_marker)
    draw_marker(draw, centroid_hires[0], centroid_hires[1], max_r)

    # ---- Act 3b: hold on converged centroid ----
    print("Act 3b: hold...")
    held = crop_resize(final_with_marker, full_box)
    for _ in range(f_hold):
        save_frame(held)

    # ---- Act 4: zoom into the final centroid (Broad Street pump) ----
    print("Act 4: zoom...")
    crop_w = 0.20 * hw
    crop_h = crop_w / aspect
    cx, cy = centroid_hires
    target_box = (
        max(0, cx - crop_w / 2),
        max(0, cy - crop_h / 2),
        min(hw, cx + crop_w / 2),
        min(hh, cy + crop_h / 2),
    )

    for i in range(f_zoom):
        t = ease_in_out((i + 1) / f_zoom)
        box = tuple(
            full_box[k] + (target_box[k] - full_box[k]) * t for k in range(4)
        )
        save_frame(crop_resize(final_with_marker, box))

    # ---- Act 5: mark the real Broad Street pump, in blue ----
    print("Act 5: pump marker...")
    pump_hires = (PUMP_NATIVE[0] * scale, PUMP_NATIVE[1] * scale)
    max_r_pump = 30 * scale
    for i in range(f_pump):
        t = ease_in_out((i + 1) / f_pump)
        canvas = final_with_marker.copy()
        draw = ImageDraw.Draw(canvas)
        draw_pump_marker(draw, pump_hires[0], pump_hires[1], max_r_pump * t)
        save_frame(crop_resize(canvas, target_box))

    final_with_both = final_with_marker.copy()
    draw = ImageDraw.Draw(final_with_both)
    draw_pump_marker(draw, pump_hires[0], pump_hires[1], max_r_pump)

    # ---- Act 6: end hold, both markers visible ----
    print("Act 6: end hold...")
    zoomed = crop_resize(final_with_both, target_box)
    for _ in range(f_endhold):
        save_frame(zoomed)

    print(f"Total frames: {frame_idx}")

    # ---- encode video ----
    print("Encoding animation.mp4...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", os.path.join(FRAME_DIR, "f_%05d.png"),
            "-c:v", "libx264", "-pix_fmt", "yuv420p",
            "-vf", "format=yuv420p",
            "animation.mp4",
        ],
        check=True,
        capture_output=True,
    )

    print("Encoding preview.gif...")
    subprocess.run(
        [
            "ffmpeg", "-y", "-framerate", str(FPS),
            "-i", os.path.join(FRAME_DIR, "f_%05d.png"),
            "-vf", "fps=15,scale=640:-1:flags=lanczos",
            "-loop", "0",
            "preview.gif",
        ],
        check=True,
        capture_output=True,
    )

    shutil.rmtree(FRAME_DIR)
    print("Done: animation.mp4, preview.gif")


if __name__ == "__main__":
    main()
