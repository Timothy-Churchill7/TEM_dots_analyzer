#!/usr/bin/env python3
"""
TEM Quantum Dot Analyzer
Detects dark quantum dot particles in TEM TIF images, measures their diameter,
and exports a CSV with dot number and length (equivalent diameter in nm and px).

Only dots that pass a shape-validation check are counted: clumps, touching
aggregates, chains, and otherwise ambiguous regions are silently excluded.
With --annotate, counted dots are drawn in green and excluded ones in red.

Usage:
    python3 tem_dot_analyzer.py YS17_016.tif
    python3 tem_dot_analyzer.py YS17_016.tif -o results.csv
    python3 tem_dot_analyzer.py YS17_016.tif --scale 0.0571  # manual nm/px
    python3 tem_dot_analyzer.py YS17_016.tif --min-size 2 --max-size 15

Both a CSV and a numbered annotated PNG are always written.
The dot numbers in the PNG match the dot_number column in the CSV.
"""

import sys
import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_fill_holes
from skimage.feature import blob_log
from skimage import measure, morphology, color

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Scale-bar auto-detection
# ---------------------------------------------------------------------------

def detect_info_bar_row(arr):
    """
    Find the first row of the data/annotation bar (bright strip) at the
    bottom of TEM images.  Returns the row index, or None.
    """
    height, width = arr.shape
    search_start = int(height * 0.80)
    right_quarter = arr[search_start:, int(width * 0.75):]
    row_means = right_quarter.mean(axis=1)
    bright = np.where(row_means > 190)[0]
    if len(bright) == 0:
        return None
    first_bright = search_start + int(bright[0])
    for row in range(first_bright, max(first_bright - 20, search_start) - 1, -1):
        if arr[row, int(width * 0.75):].mean() < 50:
            return row
    return first_bright


def detect_scale_bar_pixels(arr, info_bar_row, dark_thr=45, gap_tol=3):
    """
    Within the info bar, find the longest horizontal run of dark pixels
    (the scale bar) and return its pixel length.  Returns None if not found.

    Runs are measured with a small gap tolerance so that anti-aliasing from
    image resampling (which turns solid-black bar pixels into values of 1-40)
    does not fragment the bar.  Best run this on the full-resolution image,
    where the bar is cleanest.
    """
    if info_bar_row is None:
        return None
    height = arr.shape[0]
    best_length = 0
    for i in range(info_bar_row, height):
        row = arr[i]
        # The scale bar sits on a bright info bar / white box; skip rows that
        # have no bright background (avoids matching dark image content).
        if np.count_nonzero(row > 200) < 50:
            continue
        run = best_run = gap = 0
        for v in row:
            if v <= dark_thr:
                run += 1
                gap = 0
                if run > best_run:
                    best_run = run
            else:
                gap += 1
                if gap <= gap_tol and run > 0:
                    run += 1          # bridge a small bright gap
                else:
                    run = 0
        if best_run > best_length:
            best_length = best_run
    return int(best_length) if best_length > 20 else None


# ---------------------------------------------------------------------------
# LoG blob detection
# ---------------------------------------------------------------------------

def detect_blobs(arr, min_sigma, max_sigma, threshold, overlap=0.5, denoise=1.5, num_sigma=10):
    """
    Detect dark circular blobs via Laplacian-of-Gaussian on the inverted image.
    Returns array of shape (N, 3): [row, col, sigma].
    """
    smoothed = gaussian_filter(arr.astype(np.float32) / 255.0, sigma=denoise)
    inverted = 1.0 - smoothed
    return blob_log(
        inverted,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        num_sigma=num_sigma,
        threshold=threshold,
        overlap=overlap,
    )


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------

def validate_blob_shape(arr, y, x, sigma,
                        min_circularity=0.65,
                        max_aspect_ratio=2.0,
                        max_area_ratio=2.5):
    """
    Verify that the detected blob is a single, roughly circular dot rather
    than a clump, chain, or other ambiguous feature.

    Strategy:
      1. Extract a local patch (4r radius) around the blob.
      2. Apply heavy Gaussian smoothing (sigma = r/2.5) to wash out shot noise
         while preserving the dot's ~2r-scale structure.
      3. Threshold at the midpoint between the local background and the darkest
         pixel within r of the centre.
      4. Measure the resulting binary region with skimage.measure.regionprops.
      5. Reject if circularity < min_circularity, aspect ratio > max_aspect_ratio,
         or the region area is more than max_area_ratio × π r² (clump).

    Returns (is_valid, metrics_dict).
    """
    h, w = arr.shape
    r = sigma * np.sqrt(2)
    margin = int(r * 4) + 5

    yi, xi = int(round(y)), int(round(x))
    y0, y1 = max(0, yi - margin), min(h, yi + margin)
    x0, x1 = max(0, xi - margin), min(w, xi + margin)
    patch = arr[y0:y1, x0:x1].astype(np.float32)
    if patch.size < 9:
        return False, {}

    # Heavy smoothing: suppresses granular noise, keeps dot structure
    smooth_sigma = max(2.0, r / 2.5)
    patch_sm = gaussian_filter(patch, sigma=smooth_sigma)

    cy, cx = yi - y0, xi - x0
    yy, xx = np.ogrid[:patch_sm.shape[0], :patch_sm.shape[1]]
    dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    bg_mask = dist > r * 2.0
    inner_mask = dist <= r * 1.5
    if bg_mask.sum() == 0 or inner_mask.sum() == 0:
        return False, {}

    bg = np.median(patch_sm[bg_mask])
    core = patch_sm[inner_mask].min()
    if bg <= core:
        # No meaningful contrast — dot is indistinguishable from background
        return False, {}

    # Midpoint threshold isolates the dark dot region
    thresh = (bg + core) / 2.0
    binary = patch_sm < thresh
    binary = binary_fill_holes(binary)

    # Remove tiny noise fragments (< 10 % of expected single-dot area)
    min_obj = max(4, int(np.pi * r ** 2 * 0.10))
    binary = morphology.remove_small_objects(binary, min_size=min_obj + 1)

    labeled = measure.label(binary)
    cy_c = min(max(cy, 0), labeled.shape[0] - 1)
    cx_c = min(max(cx, 0), labeled.shape[1] - 1)
    lbl = labeled[cy_c, cx_c]

    # If centre is background, search within r pixels
    if lbl == 0:
        found = False
        step = max(1, int(r / 4))
        for dy2 in range(-int(r), int(r) + 1, step):
            for dx2 in range(-int(r), int(r) + 1, step):
                ny = min(max(cy_c + dy2, 0), labeled.shape[0] - 1)
                nx = min(max(cx_c + dx2, 0), labeled.shape[1] - 1)
                if labeled[ny, nx] > 0:
                    lbl = labeled[ny, nx]
                    found = True
                    break
            if found:
                break
    if lbl == 0:
        return False, {}

    props = measure.regionprops((labeled == lbl).astype(int))
    if not props:
        return False, {}
    p = props[0]

    circularity = (4 * np.pi * p.area) / p.perimeter ** 2 if p.perimeter > 0 else 0
    aspect_ratio = (p.axis_major_length / p.axis_minor_length
                    if p.axis_minor_length > 0 else 999.0)
    area_ratio = p.area / (np.pi * r ** 2)

    is_valid = (
        circularity >= min_circularity
        and aspect_ratio <= max_aspect_ratio
        and area_ratio <= max_area_ratio
    )

    # Equivalent diameter of the measured region (more accurate than LoG sigma)
    measured_diam_px = float(p.equivalent_diameter_area) if hasattr(p, "equivalent_diameter_area") \
        else float(np.sqrt(4 * p.area / np.pi))

    # Confidence: product of three independent shape quality scores (0–1 each).
    # circularity: 1 = perfect circle
    # 1/aspect_ratio: 1 = round, lower = elongated
    # 1/(1+|area_ratio-1|): peaks at area_ratio=1 (exactly expected dot size)
    confidence = round(
        circularity
        * min(1.0, 1.0 / max(aspect_ratio, 1.0))
        * min(1.0, 1.0 / (1.0 + abs(area_ratio - 1.0))),
        4,
    )

    return is_valid, {
        "circularity": round(circularity, 3),
        "aspect_ratio": round(aspect_ratio, 3),
        "area_ratio": round(area_ratio, 3),
        "measured_diam_px": measured_diam_px,
        "confidence": confidence,
    }


# ---------------------------------------------------------------------------
# Filter and measure accepted blobs
# ---------------------------------------------------------------------------

def process_blobs(raw_blobs, arr, info_bar_row, nm_per_pixel,
                  min_circularity, max_aspect_ratio, edge_margin=20):
    """
    Apply position and shape filters to raw LoG blobs.
    Accepted blobs are sorted by confidence score (highest first) before
    numbering, so dot #1 is always the clearest single particle.
    Returns (accepted_blobs, rejected_blobs, records).
    """
    h, w = arr.shape
    rejected_blobs = []
    candidates = []   # (y, x, sigma, metrics) — sorted later

    # Spatial pre-sort (top→bottom, left→right) is only used to break ties
    # in position; final numbering is by confidence.
    def spatial_key(b):
        y, x, _ = b
        return (int(y // 50), x)

    for blob in sorted(raw_blobs, key=spatial_key):
        y, x, sigma = blob

        # ── Position filter ──────────────────────────────────────────────
        if info_bar_row is not None and y >= info_bar_row - edge_margin:
            continue
        if y < edge_margin or x < edge_margin or x >= w - edge_margin:
            continue

        # ── Shape validation ─────────────────────────────────────────────
        is_valid, metrics = validate_blob_shape(
            arr, y, x, sigma,
            min_circularity=min_circularity,
            max_aspect_ratio=max_aspect_ratio,
        )

        if is_valid:
            candidates.append((y, x, sigma, metrics))
        else:
            rejected_blobs.append((y, x, sigma))

    # Sort accepted candidates by confidence score, highest first.
    # Dot #1 will be the most circle-like, well-isolated single particle.
    candidates.sort(key=lambda c: c[3].get("confidence", 0.0), reverse=True)

    accepted_blobs = []
    records = []
    for n, (y, x, sigma, metrics) in enumerate(candidates, start=1):
        accepted_blobs.append((y, x, sigma))
        best_diam_px = metrics.get("measured_diam_px", sigma * np.sqrt(2) * 2.0)
        rec = {
            "dot_number": n,
            "confidence": metrics.get("confidence", 0.0),
            "centroid_x_px": round(float(x), 1),
            "centroid_y_px": round(float(y), 1),
            "length_px": round(best_diam_px, 2),
        }
        if nm_per_pixel is not None:
            rec["length_nm"] = round(best_diam_px * nm_per_pixel, 3)
        records.append(rec)

    return accepted_blobs, rejected_blobs, records


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

def write_csv(records, out_path, include_nm):
    if not records:
        print("  No dots passed validation — CSV not written.")
        return
    fields = (
        ["dot_number", "confidence", "length_nm", "length_px", "centroid_x_px", "centroid_y_px"]
        if include_nm else
        ["dot_number", "confidence", "length_px", "centroid_x_px", "centroid_y_px"]
    )
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(records)
    print(f"  Saved {len(records)} dots → {out_path}")


# ---------------------------------------------------------------------------
# Annotated image
# ---------------------------------------------------------------------------

def _get_font(size):
    """Return a PIL ImageFont at the requested size, falling back gracefully."""
    from PIL import ImageFont
    candidates = [
        # macOS
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/HelveticaNeue.ttc",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        # Linux
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        # Windows
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except (IOError, OSError):
            continue
    # Pillow ≥ 10 supports a size argument on the built-in font
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def save_annotated_image(arr, accepted_blobs, rejected_blobs, out_path, info_bar_row):
    """
    Save the grayscale image with translucent number labels for accepted dots.
    Rejected blobs are not shown. Circles are rendered as an SVG overlay in the
    web UI so they can be toggled interactively.
    """
    from PIL import ImageDraw

    rgb = color.gray2rgb(arr).astype(np.uint8)
    pil_img = Image.fromarray(rgb).convert("RGBA")

    # Draw labels onto a transparent overlay so they composite at reduced opacity
    overlay = Image.new("RGBA", pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    font_size = max(14, int(arr.shape[0] / 90))
    font = _get_font(font_size)

    for dot_number, (y, x, sigma) in enumerate(accepted_blobs, start=1):
        label = str(dot_number)
        xi, yi = int(x), int(y)

        bbox = draw.textbbox((0, 0), label, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        tx = max(0, min(xi - tw // 2, arr.shape[1] - tw - 1))
        ty = max(0, min(yi - th // 2, arr.shape[0] - th - 1))

        # White fill (82% opacity), dark stroke (60% opacity)
        draw.text((tx, ty), label, font=font,
                  fill=(255, 255, 255, 210),
                  stroke_width=2,
                  stroke_fill=(0, 0, 0, 153))

    pil_img = Image.alpha_composite(pil_img, overlay).convert("RGB")
    pil_img.save(str(out_path))
    print(f"  Saved annotated image → {out_path}")
    print(f"    Labelled: {len(accepted_blobs)}  |  "
          f"Rejected (not shown): {len(rejected_blobs)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Detect and measure single quantum dots in TEM TIF images.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input", help="TIF file path")
    p.add_argument("-o", "--output", default=None, help="Output CSV path")
    p.add_argument("--scale", type=float, default=None,
                   help="nm/pixel (overrides auto scale-bar detection)")
    p.add_argument("--scale-nm", type=float, default=None,
                   help="Physical length (nm) the scale bar represents")
    p.add_argument("--no-nm", action="store_true",
                   help="Report pixel sizes only, skip nm conversion")
    p.add_argument("--min-size", type=float, default=2.0,
                   help="Minimum dot diameter in nm")
    p.add_argument("--max-size", type=float, default=15.0,
                   help="Maximum dot diameter in nm")
    p.add_argument("--threshold", type=float, default=0.12,
                   help="LoG detection threshold: lower = more (noisier) candidates")
    p.add_argument("--overlap", type=float, default=0.5,
                   help="Max fractional overlap between blobs (0–1)")
    p.add_argument("--denoise", type=float, default=1.5,
                   help="Gaussian denoise sigma (pixels) before blob detection")
    p.add_argument("--min-circularity", type=float, default=0.65,
                   help="Minimum circularity to accept a blob (0=any, 1=perfect circle)")
    p.add_argument("--max-aspect-ratio", type=float, default=2.0,
                   help="Maximum axis ratio (major/minor) to accept a blob")
    args = p.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"Error: not found: {in_path}")
        sys.exit(1)
    out_path = Path(args.output) if args.output else in_path.with_suffix(".csv")
    ann_path = out_path.with_suffix("").with_suffix(".annotated.png")

    # ── Load ────────────────────────────────────────────────────────────
    print(f"\nFile: {in_path.name}")
    img = Image.open(str(in_path))
    arr = np.array(img)
    if arr.ndim == 3:
        arr = np.mean(arr[..., :3], axis=2).astype(np.uint8)
    print(f"  Size: {arr.shape[1]}×{arr.shape[0]} px")

    # ── Scale calibration ────────────────────────────────────────────────
    nm_per_pixel = None
    info_bar_row = detect_info_bar_row(arr)

    if args.no_nm:
        print("  Scale: pixels only")
    elif args.scale is not None:
        nm_per_pixel = args.scale
        print(f"  Scale: {nm_per_pixel:.4f} nm/px (manual)")
    else:
        if info_bar_row is not None:
            bar_px = detect_scale_bar_pixels(arr, info_bar_row)
            if bar_px:
                if args.scale_nm is not None:
                    scale_nm = args.scale_nm
                else:
                    bar_fraction = bar_px / arr.shape[1]
                    scale_nm = 40.0 if bar_fraction > 0.3 else 100.0
                    print(f"  Note: scale bar nm inferred as {scale_nm} nm — "
                          f"use --scale-nm to override")
                nm_per_pixel = scale_nm / bar_px
                print(f"  Scale bar: {bar_px} px = {scale_nm} nm "
                      f"→ {nm_per_pixel:.4f} nm/px  ({1/nm_per_pixel:.1f} px/nm)")
            else:
                print("  Scale bar not found — use --scale nm/px to calibrate")
        else:
            print("  Info bar not found — use --scale nm/px to calibrate")

    # ── Mask info bar ─────────────────────────────────────────────────────
    analysis = arr.copy()
    if info_bar_row is not None:
        analysis[info_bar_row:, :] = int(arr[:info_bar_row].mean())

    # ── Sigma range ───────────────────────────────────────────────────────
    if nm_per_pixel and not args.no_nm:
        px_per_nm = 1.0 / nm_per_pixel
        min_r_px = (args.min_size * px_per_nm) / 2.0
        max_r_px = (args.max_size * px_per_nm) / 2.0
    else:
        min_r_px, max_r_px = 5.0, 150.0

    min_sigma = min_r_px / np.sqrt(2)
    max_sigma = max_r_px / np.sqrt(2)

    if nm_per_pixel and not args.no_nm:
        print(f"  Size range: {args.min_size}–{args.max_size} nm  "
              f"({min_r_px*2:.0f}–{max_r_px*2:.0f} px diam)")
    else:
        print(f"  Size range: {min_r_px*2:.0f}–{max_r_px*2:.0f} px diam")

    # ── Detect candidates ─────────────────────────────────────────────────
    print(f"  Detecting blob candidates (LoG, threshold={args.threshold}) …")
    raw_blobs = detect_blobs(
        analysis,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        threshold=args.threshold,
        overlap=args.overlap,
        denoise=args.denoise,
    )
    print(f"  {len(raw_blobs)} candidates found")

    # ── Shape validation ──────────────────────────────────────────────────
    print(f"  Validating shapes "
          f"(min_circularity={args.min_circularity}, "
          f"max_aspect_ratio={args.max_aspect_ratio}) …")
    accepted, rejected, records = process_blobs(
        raw_blobs, analysis, info_bar_row,
        nm_per_pixel=nm_per_pixel if not args.no_nm else None,
        min_circularity=args.min_circularity,
        max_aspect_ratio=args.max_aspect_ratio,
    )
    n_pos = len(raw_blobs)
    n_edge = n_pos - len(accepted) - len(rejected)
    print(f"  Accepted: {len(accepted)}  |  "
          f"Rejected (uncertain/clump): {len(rejected)}  |  "
          f"Outside margin: {n_edge}")

    # ── Summary ───────────────────────────────────────────────────────────
    if records:
        sizes_key = "length_nm" if (nm_per_pixel and not args.no_nm) else "length_px"
        unit = "nm" if sizes_key == "length_nm" else "px"
        sizes = [r[sizes_key] for r in records]
        print(f"\n  Diameter summary ({unit}):")
        print(f"    N    = {len(sizes)}")
        print(f"    Mean = {np.mean(sizes):.2f} {unit}")
        print(f"    Std  = {np.std(sizes):.2f} {unit}")
        print(f"    Min  = {np.min(sizes):.2f} {unit}")
        print(f"    Max  = {np.max(sizes):.2f} {unit}")

    # ── Output ────────────────────────────────────────────────────────────
    write_csv(records, out_path, nm_per_pixel is not None and not args.no_nm)
    save_annotated_image(arr, accepted, rejected, ann_path, info_bar_row)


if __name__ == "__main__":
    main()
