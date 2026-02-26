#!/usr/bin/env python3
"""
Temporal Inference Pipeline
===========================
Run the trained DDT model across multi-year imagery (2002-2023) with
histogram matching to normalize color/brightness differences between years.

Steps:
    1. Build reference color profile from 2020 training image
    2. For each year's image:
       a. Apply per-band histogram matching to reference
       b. Run streaming inference → predicted DTM raster
       c. Watershed segmentation → crown GeoPackage
    3. Generate per-year summary statistics

Usage:
    python scripts/temporal_inference.py --config configs/colab.yaml \
        --temporal_dir /content/drive/MyDrive/treedata/temporal_imagery \
        --output_dir /content/drive/MyDrive/treedata/temporal_results \
        --years 2023 2021 2019 2017 2015 2013 2012 2009 2007 2005 2002

    # Skip histogram matching (run raw)
    python scripts/temporal_inference.py --config configs/colab.yaml \
        --temporal_dir /path/to/imagery --no_histmatch

    # Run only specific years
    python scripts/temporal_inference.py --config configs/colab.yaml \
        --temporal_dir /path/to/imagery --years 2023 2019 2013

Author:  Kam (Edmonds Climate Advisory Board)
License: MIT
"""

import argparse
import gc
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import albumentations as A
import fiona
import fiona.crs
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.windows
import segmentation_models_pytorch as smp
import torch
import yaml
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import label as scipy_label
from shapely.geometry import mapping, shape
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from tqdm import tqdm

warnings.filterwarnings("ignore")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Available years and their zoom levels
YEAR_INFO = {
    2002: {"zoom": 19, "ground_res_cm": 20.1},
    2005: {"zoom": 19, "ground_res_cm": 20.1},
    2007: {"zoom": 19, "ground_res_cm": 20.1},
    2009: {"zoom": 19, "ground_res_cm": 20.1},
    2012: {"zoom": 19, "ground_res_cm": 20.1},
    2013: {"zoom": 20, "ground_res_cm": 10.0},
    2015: {"zoom": 20, "ground_res_cm": 10.0},
    2017: {"zoom": 20, "ground_res_cm": 10.0},
    2019: {"zoom": 20, "ground_res_cm": 10.0},
    2021: {"zoom": 20, "ground_res_cm": 10.0},
    2023: {"zoom": 20, "ground_res_cm": 10.0},
}


# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    "encoder": "resnet101",
    "decoder_channels": [1024, 512, 256, 128, 64],
    "checkpoint_dir": "checkpoints",
    "tile_size": 512,
    "infer_stride": 256,
    "infer_batch_size": 160,
    "chunk_size": 2048,
    "chunk_border": 300,
    "min_distance": 30,
    "dtm_threshold": 10.0,
    "min_crown_area_m2": 2.0,
}


def load_config(config_path=None):
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
        if user_cfg:
            cfg.update(user_cfg)
    return cfg


# ═══════════════════════════════════════════════════════════════════
# HISTOGRAM MATCHING
# ═══════════════════════════════════════════════════════════════════

def build_reference_profile(ref_path, n_samples=300, chip_size=512, seed=42):
    """Sample random chips from the 2020 reference image to build per-band
    cumulative distribution functions (CDFs) for histogram matching.

    Returns:
        ref_cdfs: list of 3 arrays, each shape (256,) — the CDF for R, G, B
        ref_means: per-band means for quick QA
    """
    print(f"\n── Building reference color profile from 2020 image ──")
    print(f"  Sampling {n_samples} × {chip_size}×{chip_size} chips...")

    rng = np.random.default_rng(seed)
    band_pixels = [[], [], []]  # collect pixel values per band

    with rasterio.open(ref_path) as src:
        h, w = src.height, src.width
        sampled = 0

        for _ in range(n_samples * 2):  # over-sample to handle skips
            if sampled >= n_samples:
                break
            x = rng.integers(0, max(1, w - chip_size))
            y = rng.integers(0, max(1, h - chip_size))
            win = rasterio.windows.Window(x, y, chip_size, chip_size)
            chip = src.read(window=win)  # (3, H, W)

            # Skip black/empty areas (nodata, coverage gaps)
            if chip.mean() < 10:
                continue

            for b in range(3):
                # Subsample pixels within chip to keep memory reasonable
                flat = chip[b].ravel()
                idx = rng.choice(len(flat), size=min(5000, len(flat)), replace=False)
                band_pixels[b].append(flat[idx])
            sampled += 1

    # Build CDFs
    ref_cdfs = []
    ref_means = []
    for b in range(3):
        all_vals = np.concatenate(band_pixels[b])
        ref_means.append(float(all_vals.mean()))

        # Histogram → CDF
        hist, _ = np.histogram(all_vals, bins=256, range=(0, 256))
        cdf = hist.cumsum().astype(np.float64)
        cdf = cdf / cdf[-1]  # normalize to [0, 1]
        ref_cdfs.append(cdf)

    print(f"  Sampled {sampled} chips")
    print(f"  Reference means: R={ref_means[0]:.1f} G={ref_means[1]:.1f} B={ref_means[2]:.1f}")

    return ref_cdfs, ref_means


def build_source_profile(src_path, n_samples=200, chip_size=512, seed=42):
    """Sample the source (year) image to build its per-band CDF."""
    rng = np.random.default_rng(seed)
    band_pixels = [[], [], []]

    with rasterio.open(src_path) as src:
        h, w = src.height, src.width
        sampled = 0

        for _ in range(n_samples * 2):
            if sampled >= n_samples:
                break
            x = rng.integers(0, max(1, w - chip_size))
            y = rng.integers(0, max(1, h - chip_size))
            win = rasterio.windows.Window(x, y, chip_size, chip_size)
            chip = src.read(window=win)

            if chip.mean() < 10:
                continue

            for b in range(3):
                flat = chip[b].ravel()
                idx = rng.choice(len(flat), size=min(5000, len(flat)), replace=False)
                band_pixels[b].append(flat[idx])
            sampled += 1

    src_cdfs = []
    src_means = []
    for b in range(3):
        all_vals = np.concatenate(band_pixels[b])
        src_means.append(float(all_vals.mean()))
        hist, _ = np.histogram(all_vals, bins=256, range=(0, 256))
        cdf = hist.cumsum().astype(np.float64)
        cdf = cdf / cdf[-1]
        src_cdfs.append(cdf)

    return src_cdfs, src_means


def compute_lut(src_cdf, ref_cdf):
    """Compute a 256-entry lookup table that maps source pixel values to
    reference pixel values via CDF matching.

    For each source value s, find the reference value r where
    ref_cdf[r] is closest to src_cdf[s].
    """
    lut = np.zeros(256, dtype=np.uint8)
    for s in range(256):
        # Find closest match in reference CDF
        idx = np.searchsorted(ref_cdf, src_cdf[s])
        lut[s] = np.clip(idx, 0, 255)
    return lut


def apply_lut_to_tile(tile, luts):
    """Apply per-band LUTs to a (3, H, W) uint8 tile. Fast vectorized."""
    out = np.empty_like(tile)
    for b in range(3):
        out[b] = luts[b][tile[b]]
    return out


# ═══════════════════════════════════════════════════════════════════
# WATERSHED (same as pipeline.py)
# ═══════════════════════════════════════════════════════════════════

def _process_watershed_chunk(args):
    """Worker: watershed on one spatial chunk of the DTM."""
    (r0, c0, dtm_path, chunk_size, chunk_border, img_h, img_w,
     min_dist, dtm_thresh, min_area, px_area, px_x, px_y, bounds) = args

    r0b = max(0, r0 - chunk_border)
    c0b = max(0, c0 - chunk_border)
    r1b = min(img_h, r0 + chunk_size + chunk_border)
    c1b = min(img_w, c0 + chunk_size + chunk_border)

    with rasterio.open(dtm_path) as src:
        win = rasterio.windows.Window(c0b, r0b, c1b - c0b, r1b - r0b)
        chunk = src.read(1, window=win)

    mask = chunk >= dtm_thresh
    if mask.sum() == 0:
        return []

    coords = peak_local_max(chunk, min_distance=min_dist,
                            labels=mask, threshold_abs=dtm_thresh)
    if len(coords) == 0:
        return []

    seeds = np.zeros(chunk.shape, dtype=bool)
    seeds[tuple(coords.T)] = True
    markers, _ = scipy_label(seeds)
    ws = watershed(-chunk, markers, mask=mask)

    records = []
    for cid in np.unique(ws):
        if cid == 0:
            continue
        px = ws == cid
        area = px.sum() * px_area
        if area < min_area:
            continue

        rows, cols = np.where(px)
        abs_r, abs_c = rows + r0b, cols + c0b
        cr, cc = int(abs_r.mean()), int(abs_c.mean())

        # Centroid ownership: skip if centroid is in overlap zone
        if cr < r0b + chunk_border and r0b > 0:
            continue
        if cr >= r1b - chunk_border and r1b < img_h:
            continue
        if cc < c0b + chunk_border and c0b > 0:
            continue
        if cc >= c1b - chunk_border and c1b < img_w:
            continue

        r_min, r_max = int(abs_r.min()), int(abs_r.max())
        c_min, c_max = int(abs_c.min()), int(abs_c.max())
        lh, lw = r_max - r_min + 1, c_max - c_min + 1

        local_mask = np.zeros((lh, lw), dtype=np.uint8)
        local_mask[abs_r - r_min, abs_c - c_min] = 1

        local_tf = rasterio.transform.from_bounds(
            bounds.left + c_min * px_x, bounds.top - (r_max + 1) * px_y,
            bounds.left + (c_max + 1) * px_x, bounds.top - r_min * px_y,
            lw, lh)

        geoms = [shape(g) for g, v in
                 rasterio.features.shapes(local_mask, transform=local_tf) if v == 1]
        if not geoms:
            continue
        geom = geoms[0] if len(geoms) == 1 else geoms[0].union(*geoms[1:])

        sc = "small" if area <= 4.9 else ("medium" if area <= 15.9 else "large")
        records.append({
            "geometry": mapping(geom), "area_m2": round(float(area), 2),
            "diameter_m": round(2 * (area / 3.14159) ** 0.5, 2),
            "size_class": sc,
        })
    return records


# ═══════════════════════════════════════════════════════════════════
# INFERENCE FOR ONE YEAR
# ═══════════════════════════════════════════════════════════════════

def infer_one_year(year, img_path, model, cfg, output_dir,
                   luts=None, device=None):
    """Run streaming inference + watershed on a single year's image.

    Args:
        year: int, the year label
        img_path: Path to the year's GeoTIFF
        model: loaded PyTorch model (eval mode)
        cfg: config dict
        output_dir: Path for outputs
        luts: list of 3 LUT arrays for histogram matching, or None
        device: torch device
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dtm_out = output_dir / f"edmonds_{year}_dtm.tif"
    crowns_out = output_dir / f"edmonds_{year}_crowns.gpkg"
    stats_out = output_dir / f"edmonds_{year}_stats.csv"

    # Skip if already complete
    if crowns_out.exists() and stats_out.exists():
        print(f"  [{year}] Already complete, skipping. Delete {crowns_out.name} to re-run.")
        return

    tile_size = cfg["tile_size"]
    stride = cfg["infer_stride"]
    center_crop = stride
    pad = (tile_size - stride) // 2
    batch_size = cfg.get("infer_batch_size", 32)

    # Image metadata
    with rasterio.open(img_path) as src:
        h, w = src.height, src.width
        crs, tf, bounds = src.crs, src.transform, src.bounds
        px_x, px_y = tf.a, abs(tf.e)
        px_area = px_x * px_y

    print(f"\n  [{year}] Inference: {w}×{h} px | "
          f"res={px_x*100:.1f} cm | histmatch={'yes' if luts else 'no'}")

    infer_tf = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2()])

    # Build tile origins
    origins = sorted(set(
        [(r, c) for r in range(0, h, stride) for c in range(0, w, stride)] +
        [(h - tile_size, c) for c in range(0, w, stride)] +
        [(r, w - tile_size) for r in range(0, h, stride)] +
        [(h - tile_size, w - tile_size)]
    ))

    # Filter out negative origins (images smaller than tile_size)
    origins = [(r, c) for r, c in origins if r >= 0 and c >= 0]

    # If image is smaller than tile_size, handle specially
    if h < tile_size or w < tile_size:
        print(f"  [{year}] WARNING: Image smaller than tile_size ({h}×{w} < {tile_size})")
        print(f"  [{year}] Padding image to minimum size")
        origins = [(0, 0)]

    # Streaming inference → DTM
    dtm_profile = {
        "driver": "GTiff", "dtype": "float32", "width": w, "height": h,
        "count": 1, "crs": crs, "transform": tf, "compress": "lzw",
        "nodata": 0.0, "BIGTIFF": "YES",
        "tiled": True, "blockxsize": 512, "blockysize": 512,
    }

    batch_imgs, batch_coords = [], []

    def flush(imgs, coords, dst):
        if not imgs:
            return
        inp = torch.stack(imgs).to(device)
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = model(inp).squeeze(1).cpu().numpy()
        pred = np.clip(pred, 0, 100)
        for k, (ro, co) in enumerate(coords):
            c = pred[k, pad:pad + center_crop, pad:pad + center_crop]
            ch = min(ro + center_crop, h) - ro
            cw = min(co + center_crop, w) - co
            win = rasterio.windows.Window(co, ro, cw, ch)
            dst.write(c[:ch, :cw][np.newaxis].astype(np.float32), window=win)

    t0 = time.time()
    with rasterio.open(dtm_out, "w", **dtm_profile) as dst:
        with rasterio.open(img_path) as src:
            for ro, co in tqdm(origins, desc=f"  [{year}] Inference",
                               miniters=2000, leave=False):
                rs, cs = ro - pad, co - pad
                re, ce = rs + tile_size, cs + tile_size

                rr0, rc0 = max(0, rs), max(0, cs)
                rr1, rc1 = min(h, re), min(w, ce)
                win = rasterio.windows.Window(rc0, rr0, rc1 - rc0, rr1 - rr0)
                tile = src.read([1, 2, 3], window=win)  # (3, H, W)

                # Apply histogram matching LUT
                if luts is not None:
                    tile = apply_lut_to_tile(tile, luts)

                tile = tile.transpose(1, 2, 0)  # (H, W, 3)

                # Pad if needed
                pt, pb = rr0 - rs, re - rr1
                pl, pr = rc0 - cs, ce - rc1
                if any([pt, pb, pl, pr]):
                    tile = np.pad(tile, ((pt, pb), (pl, pr), (0, 0)), mode="reflect")

                batch_imgs.append(infer_tf(image=tile)["image"])
                batch_coords.append((ro, co))

                if len(batch_imgs) == batch_size:
                    flush(batch_imgs, batch_coords, dst)
                    batch_imgs, batch_coords = [], []

            flush(batch_imgs, batch_coords, dst)

    infer_time = time.time() - t0
    dtm_size_mb = dtm_out.stat().st_size / 1e6
    print(f"  [{year}] DTM complete: {dtm_size_mb:.0f} MB in {infer_time:.0f}s")

    # Watershed → crowns
    print(f"  [{year}] Watershed segmentation...")
    t0 = time.time()
    n_workers = min(8, multiprocessing.cpu_count())
    chunk_args = [
        (r0, c0, str(dtm_out), cfg["chunk_size"], cfg["chunk_border"],
         h, w, cfg["min_distance"], cfg["dtm_threshold"],
         cfg["min_crown_area_m2"], px_area, px_x, px_y, bounds)
        for r0 in range(0, h, cfg["chunk_size"])
        for c0 in range(0, w, cfg["chunk_size"])
    ]

    schema = {"geometry": "Polygon", "properties": {
        "crown_id": "str", "area_m2": "float",
        "diameter_m": "float", "size_class": "str"}}

    crown_n = 0
    with fiona.open(crowns_out, "w", driver="GPKG",
                    crs=crs.to_wkt(), schema=schema) as gpkg:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futs = {pool.submit(_process_watershed_chunk, a): a for a in chunk_args}
            for fut in tqdm(as_completed(futs), total=len(chunk_args),
                            desc=f"  [{year}] Watershed", mininterval=15, leave=False):
                for rec in fut.result():
                    gpkg.writerecords([{
                        "geometry": rec["geometry"],
                        "properties": {
                            "crown_id": f"EDM_{year}_{crown_n:07d}",
                            "area_m2": rec["area_m2"],
                            "diameter_m": rec["diameter_m"],
                            "size_class": rec["size_class"],
                        }}])
                    crown_n += 1

    ws_time = time.time() - t0
    print(f"  [{year}] Crowns: {crown_n:,} in {ws_time:.0f}s")

    # Stats
    gdf = gpd.read_file(crowns_out)
    if len(gdf) > 0:
        total_ha = gdf["area_m2"].sum() / 10_000
        cover_pct = 100 * gdf["area_m2"].sum() / (w * h * px_area)
        median_area = gdf["area_m2"].median()
        median_diam = gdf["diameter_m"].median()
        large_count = (gdf["size_class"] == "large").sum()
        medium_count = (gdf["size_class"] == "medium").sum()
        small_count = (gdf["size_class"] == "small").sum()
    else:
        total_ha = cover_pct = median_area = median_diam = 0
        large_count = medium_count = small_count = 0

    stats = pd.DataFrame([
        {"metric": "year", "value": year},
        {"metric": "image_width_px", "value": w},
        {"metric": "image_height_px", "value": h},
        {"metric": "pixel_res_cm", "value": round(px_x * 100, 2)},
        {"metric": "total_crowns", "value": len(gdf)},
        {"metric": "large_crowns", "value": large_count},
        {"metric": "medium_crowns", "value": medium_count},
        {"metric": "small_crowns", "value": small_count},
        {"metric": "total_canopy_ha", "value": round(total_ha, 2)},
        {"metric": "canopy_cover_pct", "value": round(cover_pct, 2)},
        {"metric": "median_area_m2", "value": round(median_area, 2)},
        {"metric": "median_diameter_m", "value": round(median_diam, 2)},
        {"metric": "inference_time_s", "value": round(infer_time, 1)},
        {"metric": "watershed_time_s", "value": round(ws_time, 1)},
        {"metric": "histogram_matched", "value": 1 if luts else 0},
    ])
    stats.to_csv(stats_out, index=False)

    print(f"  [{year}] Canopy: {total_ha:.1f} ha ({cover_pct:.1f}%) | "
          f"L={large_count:,} M={medium_count:,} S={small_count:,}")

    # Free memory
    del gdf
    gc.collect()

    return {
        "year": year, "crowns": crown_n, "canopy_ha": total_ha,
        "cover_pct": cover_pct, "infer_time": infer_time,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Temporal inference across multi-year Edmonds imagery")
    parser.add_argument("--config", type=str, default="configs/colab.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--temporal_dir", type=str, required=True,
                        help="Directory containing edmonds_YYYY_image.tif files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for per-year outputs")
    parser.add_argument("--years", type=int, nargs="+",
                        default=[2023, 2021, 2019, 2017, 2015, 2013,
                                 2012, 2009, 2007, 2005, 2002],
                        help="Years to process")
    parser.add_argument("--no_histmatch", action="store_true",
                        help="Skip histogram matching (run on raw imagery)")
    parser.add_argument("--ref_samples", type=int, default=300,
                        help="Number of chips to sample for reference profile")
    args = parser.parse_args()

    cfg = load_config(args.config)
    temporal_dir = Path(args.temporal_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  Edmonds Temporal Inference Pipeline")
    print(f"  Years: {sorted(args.years)}")
    print(f"  Device: {device}")
    print(f"  Histogram matching: {'OFF' if args.no_histmatch else 'ON'}")
    print("=" * 60)

    # ── Validate inputs ──
    ref_path = cfg.get("full_image_path")
    if not ref_path or not Path(ref_path).exists():
        print("ERROR: full_image_path not set or file not found in config.")
        print("  This is needed as the 2020 reference for histogram matching.")
        return

    ckpt_path = Path(cfg["checkpoint_dir"]) / "ddt_best_global.pt"
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    # Find available imagery
    available = {}
    for year in sorted(args.years):
        img_path = temporal_dir / f"edmonds_{year}_image.tif"
        if img_path.exists():
            available[year] = img_path
        else:
            print(f"  WARNING: {img_path.name} not found, skipping {year}")

    if not available:
        print("ERROR: No imagery files found. Check --temporal_dir path.")
        return

    print(f"\n  Found imagery for {len(available)} years: {sorted(available.keys())}")

    # ── Build reference profile ──
    ref_cdfs = None
    if not args.no_histmatch:
        ref_cdfs, ref_means = build_reference_profile(
            ref_path, n_samples=args.ref_samples)

    # ── Load model ──
    print(f"\n── Loading model from {ckpt_path.name} ──")
    model = smp.Unet(
        encoder_name=cfg["encoder"], encoder_weights=None,
        decoder_channels=tuple(cfg["decoder_channels"]),
        in_channels=3, classes=1, activation=None).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Try to compile for speed (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("  Model compiled with torch.compile()")
    except Exception:
        print("  torch.compile() not available, using eager mode")

    print(f"  Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Process each year ──
    all_results = []
    for year in sorted(available.keys(), reverse=True):  # newest first
        img_path = available[year]

        print(f"\n{'='*60}")
        print(f"  YEAR {year}")
        print(f"{'='*60}")

        # Build per-year LUTs
        luts = None
        if ref_cdfs is not None:
            print(f"  [{year}] Computing histogram matching LUTs...")
            src_cdfs, src_means = build_source_profile(str(img_path))
            luts = [compute_lut(src_cdfs[b], ref_cdfs[b]) for b in range(3)]

            # Report color shift
            print(f"  [{year}] Source means: R={src_means[0]:.1f} "
                  f"G={src_means[1]:.1f} B={src_means[2]:.1f}")

            # Quick sanity check: apply LUT to source means
            mapped = [float(luts[b][int(src_means[b])]) for b in range(3)]
            print(f"  [{year}] Mapped means: R={mapped[0]:.0f} "
                  f"G={mapped[1]:.0f} B={mapped[2]:.0f}")

        result = infer_one_year(
            year, str(img_path), model, cfg, output_dir,
            luts=luts, device=device)

        if result:
            all_results.append(result)

        # Clean up GPU memory between years
        torch.cuda.empty_cache()
        gc.collect()

    # ── Summary ──
    if all_results:
        print(f"\n{'='*60}")
        print("  TEMPORAL SUMMARY")
        print(f"{'='*60}")

        summary = pd.DataFrame(all_results).sort_values("year")
        summary_path = output_dir / "temporal_summary.csv"
        summary.to_csv(summary_path, index=False)

        print(f"\n  {'Year':<6} {'Crowns':>10} {'Canopy (ha)':>12} "
              f"{'Cover %':>8} {'Time (min)':>10}")
        print(f"  {'─'*6} {'─'*10} {'─'*12} {'─'*8} {'─'*10}")
        for _, row in summary.iterrows():
            total_time = (row.get("infer_time", 0)) / 60
            print(f"  {int(row['year']):<6} {int(row['crowns']):>10,} "
                  f"{row['canopy_ha']:>12.1f} {row['cover_pct']:>8.1f} "
                  f"{total_time:>10.1f}")

        print(f"\n  Results saved: {summary_path}")
        print(f"  Per-year outputs: {output_dir}")

    print("\n✓ Temporal inference complete.")


if __name__ == "__main__":
    main()
