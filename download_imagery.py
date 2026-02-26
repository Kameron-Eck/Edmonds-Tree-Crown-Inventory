#!/usr/bin/env python3
"""
Edmonds Temporal Imagery Downloader
=====================================
Downloads aerial imagery from King County ArcGIS tile services and
mosaics into GeoTIFFs matching the extent of the existing 2020 image.

Usage in Colab:
    from google.colab import drive
    drive.mount('/content/drive')

    !python download_imagery.py \
        --reference "/content/drive/MyDrive/treedata/Full_Image/edmonds_2020_image.tif" \
        --output_dir "/content/drive/MyDrive/treedata/temporal_imagery" \
        --years 2023 2021 2019 2017 2015 2013 \
        --workers 16

    # Older lower-res imagery:
    !python download_imagery.py \
        --reference "/content/drive/MyDrive/treedata/Full_Image/edmonds_2020_image.tif" \
        --output_dir "/content/drive/MyDrive/treedata/temporal_imagery" \
        --years 2012 2009 2007 2005 2002 \
        --workers 16

Resolution Notes:
    - 2013-2023: zoom 20 (~10 cm/pixel), 256px tiles
    - 2002-2012: zoom 19 (~20 cm/pixel), 256px tiles
    - Output CRS matches reference image (EPSG:3857)
    - Each year produces one BigTIFF with JPEG compression
"""

import argparse
import io
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

try:
    from PIL import Image
except ImportError:
    os.system("pip install Pillow -q")
    from PIL import Image

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    os.system("pip install requests -q")
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

try:
    from tqdm import tqdm
except ImportError:
    os.system("pip install tqdm -q")
    from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
# SERVICE CONFIGURATION
# ══════════════════════════════════════════════════════════════

SERVICES = {
    2023: "KingCo_Aerial_2023",
    2021: "KingCo_Aerial_2021",
    2019: "KingCo_Aerial_2019",
    2017: "KingCo_Aerial_2017",
    2015: "KingCo_Aerial_2015",
    2013: "KingCo_Aerial_2013",
    2012: "KingCo_Aerial_2012",
    2009: "KingCo_Aerial_2009",
    2007: "KingCo_Aerial_2007",
    2005: "KingCo_Aerial_2005",
    2002: "KingCo_Aerial_2002",
}

BASE_URL = "https://gismaps.kingcounty.gov/arcgis/rest/services/BaseMaps"
TILE_PX = 256

# Max zoom per year
MAX_ZOOM = {
    2023: 20, 2021: 20, 2019: 20, 2017: 20, 2015: 20, 2013: 20,
    2012: 19, 2009: 19, 2007: 19, 2005: 19, 2002: 19,
}

# Web Mercator half-circumference
ORIGIN_SHIFT = 20037508.342789244


# ══════════════════════════════════════════════════════════════
# COORDINATE / TILE MATH
# ══════════════════════════════════════════════════════════════

def tile_resolution(zoom):
    """Meters per pixel at given zoom (Web Mercator, at equator — but
    tiles are square in pixel space so this is the transform resolution)."""
    return (2 * ORIGIN_SHIFT) / (TILE_PX * 2**zoom)


def meters_to_tile(mx, my, zoom):
    """Web Mercator meters → tile (col, row)."""
    res = tile_resolution(zoom)
    col = int(math.floor((mx + ORIGIN_SHIFT) / (TILE_PX * res)))
    row = int(math.floor((ORIGIN_SHIFT - my) / (TILE_PX * res)))
    return col, row


def tile_to_meters(col, row, zoom):
    """Tile (col, row) → Web Mercator meters of top-left corner."""
    res = tile_resolution(zoom)
    mx = col * TILE_PX * res - ORIGIN_SHIFT
    my = ORIGIN_SHIFT - row * TILE_PX * res
    return mx, my


def ground_resolution_cm(zoom, lat=47.81):
    """Actual ground resolution in cm/pixel at a given latitude."""
    return 15654303.0 * math.cos(math.radians(lat)) / (2**zoom)


# ══════════════════════════════════════════════════════════════
# REFERENCE IMAGE EXTENT
# ══════════════════════════════════════════════════════════════

def get_reference_extent(ref_path):
    """Read the bounding box and metadata from the reference image."""
    with rasterio.open(ref_path) as src:
        bounds = src.bounds  # left, bottom, right, top in CRS units
        crs = src.crs
        width = src.width
        height = src.height
        res_x, res_y = src.res

    print(f"\n  Reference image: {Path(ref_path).name}")
    print(f"  Dimensions:      {width:,} × {height:,} px")
    print(f"  Pixel size:      {res_x*100:.2f} cm")
    print(f"  CRS:             {crs}")
    print(f"  Bounds (m):      W={bounds.left:.1f}  E={bounds.right:.1f}")
    print(f"                   S={bounds.bottom:.1f}  N={bounds.top:.1f}")
    print(f"  Extent:          {(bounds.right-bounds.left)/1000:.2f} × "
          f"{(bounds.top-bounds.bottom)/1000:.2f} km")

    return {
        "left": bounds.left,
        "bottom": bounds.bottom,
        "right": bounds.right,
        "top": bounds.top,
        "crs": crs,
        "ref_width": width,
        "ref_height": height,
        "ref_res": res_x,
    }


# ══════════════════════════════════════════════════════════════
# TILE GRID CALCULATION
# ══════════════════════════════════════════════════════════════

def compute_tile_grid(extent, zoom):
    """
    Compute tile grid covering the reference extent at the given zoom.

    Returns dict with grid params, actual mosaic bounds, and tile list.
    """
    # Reference bounds in Web Mercator meters
    left, bottom, right, top = extent["left"], extent["bottom"], extent["right"], extent["top"]

    # Find tile range that covers the extent
    col_min, row_min = meters_to_tile(left, top, zoom)      # NW corner
    col_max, row_max = meters_to_tile(right, bottom, zoom)  # SE corner

    n_cols = col_max - col_min + 1
    n_rows = row_max - row_min + 1
    total_tiles = n_cols * n_rows

    # Actual mosaic bounds (will be slightly larger than reference due to tile snapping)
    mos_left, mos_top = tile_to_meters(col_min, row_min, zoom)
    mos_right, mos_bottom = tile_to_meters(col_max + 1, row_max + 1, zoom)

    img_width = n_cols * TILE_PX
    img_height = n_rows * TILE_PX

    res_cm = ground_resolution_cm(zoom)

    # Build tile list
    tiles = []
    for row in range(row_min, row_max + 1):
        for col in range(col_min, col_max + 1):
            tiles.append((row, col))

    print(f"\n  Zoom level:      {zoom}")
    print(f"  Ground res:      {res_cm:.1f} cm/pixel")
    print(f"  Tile grid:       {n_cols} × {n_rows} = {total_tiles:,} tiles")
    print(f"  Mosaic size:     {img_width:,} × {img_height:,} px")
    print(f"  Mosaic extent:   {(mos_right-mos_left)/1000:.2f} × "
          f"{(mos_top-mos_bottom)/1000:.2f} km")

    est_gb = img_width * img_height * 3 / 1e9
    print(f"  Est. file size:  ~{est_gb * 0.15:.1f} GB (JPEG compressed)")

    return {
        "zoom": zoom,
        "col_min": col_min,
        "col_max": col_max,
        "row_min": row_min,
        "row_max": row_max,
        "n_cols": n_cols,
        "n_rows": n_rows,
        "total_tiles": total_tiles,
        "img_width": img_width,
        "img_height": img_height,
        "bounds": (mos_left, mos_bottom, mos_right, mos_top),
        "tiles": tiles,
    }


# ══════════════════════════════════════════════════════════════
# TILE DOWNLOADING
# ══════════════════════════════════════════════════════════════

def create_session(max_retries=5):
    """Requests session with retry logic."""
    session = requests.Session()
    retry = Retry(
        total=max_retries,
        backoff_factor=0.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Edmonds Tree Research)"
    })
    return session


def download_single_tile(session, service_name, zoom, row, col, timeout=30, max_attempts=3):
    """Download one tile with retries → (row, col, rgb_array|None)."""
    url = f"{BASE_URL}/{service_name}/MapServer/tile/{zoom}/{row}/{col}"
    for attempt in range(max_attempts):
        try:
            resp = session.get(url, timeout=timeout)
            if resp.status_code == 200 and "image" in resp.headers.get("Content-Type", ""):
                img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                return (row, col, np.array(img))
            elif resp.status_code == 404:
                # Tile doesn't exist in this service — no point retrying
                return (row, col, None)
            elif resp.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            return (row, col, None)
        except Exception:
            time.sleep(1)
            continue
    return (row, col, None)


# ══════════════════════════════════════════════════════════════
# STREAMING MOSAIC WRITER
# ══════════════════════════════════════════════════════════════

def download_and_mosaic(year, service_name, grid, output_path, workers=16):
    """
    Download all tiles and write directly to a BigTIFF using streaming
    row-by-row writes to keep memory usage low.

    Strategy:
        - Process one tile-row at a time (all columns in that row)
        - Download all tiles for the current row in parallel
        - Write TILE_PX scanlines to the output raster
        - Move to next tile-row
    """
    zoom = grid["zoom"]
    n_cols = grid["n_cols"]
    n_rows = grid["n_rows"]
    img_w = grid["img_width"]
    img_h = grid["img_height"]
    bounds = grid["bounds"]  # left, bottom, right, top

    transform = from_bounds(bounds[0], bounds[1], bounds[2], bounds[3], img_w, img_h)

    # Create output BigTIFF with JPEG compression (good for aerial RGB)
    profile = {
        "driver": "GTiff",
        "dtype": "uint8",
        "width": img_w,
        "height": img_h,
        "count": 3,
        "crs": "EPSG:3857",
        "transform": transform,
        "compress": "jpeg",
        "jpeg_quality": 90,
        "tiled": True,
        "blockxsize": 512,
        "blockysize": 512,
        "bigtiff": "yes",
        "photometric": "ycbcr",
    }

    session = create_session()
    total_downloaded = 0
    total_missing = 0  # 404s — tile doesn't exist in service
    total_failed = 0   # actual errors

    print(f"\n  Writing: {output_path}")
    print(f"  Processing {n_rows} tile-rows × {n_cols} tile-cols...")

    BATCH_SIZE = 20  # Download this many tiles at a time within a row

    with rasterio.open(output_path, "w", **profile) as dst:
        for tile_row_idx, tile_row in enumerate(
            range(grid["row_min"], grid["row_max"] + 1)
        ):
            # Assemble strip for this tile-row
            strip = np.zeros((TILE_PX, img_w, 3), dtype=np.uint8)

            # Process columns in batches to avoid overwhelming server
            all_cols = list(range(grid["col_min"], grid["col_max"] + 1))

            for batch_start in range(0, len(all_cols), BATCH_SIZE):
                batch_cols = all_cols[batch_start:batch_start + BATCH_SIZE]

                with ThreadPoolExecutor(max_workers=min(workers, len(batch_cols))) as executor:
                    futures = {
                        executor.submit(
                            download_single_tile, session, service_name,
                            zoom, tile_row, c
                        ): c
                        for c in batch_cols
                    }
                    for future in as_completed(futures):
                        r, c, arr = future.result()
                        if arr is not None:
                            col_idx = c - grid["col_min"]
                            x_start = col_idx * TILE_PX
                            h_px = min(arr.shape[0], TILE_PX)
                            w_px = min(arr.shape[1], TILE_PX)
                            strip[:h_px, x_start:x_start + w_px, :] = arr[:h_px, :w_px, :]
                            total_downloaded += 1
                        else:
                            total_missing += 1

                # Small pause between batches
                time.sleep(0.2)

            # Write strip to raster
            window = rasterio.windows.Window(0, tile_row_idx * TILE_PX, img_w, TILE_PX)
            for band in range(3):
                dst.write(strip[:, :, band], band + 1, window=window)

            # Progress
            pct = 100 * (tile_row_idx + 1) / n_rows
            print(f"\r  [{year}] Row {tile_row_idx+1}/{n_rows} "
                  f"({pct:.0f}%) | "
                  f"OK: {total_downloaded:,} | "
                  f"No tile: {total_missing:,}", end="", flush=True)

    pct_ok = 100 * total_downloaded / (total_downloaded + total_missing) if (total_downloaded + total_missing) > 0 else 0
    print(f"\n  ✓ Complete: {total_downloaded:,} tiles downloaded, "
          f"{total_missing:,} outside coverage ({pct_ok:.0f}% coverage)")

    # Report file size
    size_gb = Path(output_path).stat().st_size / 1e9
    print(f"  File size: {size_gb:.2f} GB")

    return {
        "year": year,
        "tiles_downloaded": total_downloaded,
        "tiles_failed": total_failed,
        "file_size_gb": round(size_gb, 2),
        "dimensions": f"{img_w}×{img_h}",
        "resolution_cm": round(ground_resolution_cm(zoom), 1),
    }


# ══════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_output(output_path, extent):
    """Quick validation of the output raster."""
    with rasterio.open(output_path) as src:
        print(f"\n  Validation: {Path(output_path).name}")
        print(f"  Dimensions:  {src.width:,} × {src.height:,} px")
        print(f"  CRS:         {src.crs}")
        print(f"  Pixel size:  {src.res[0]*100:.2f} cm")
        print(f"  Bands:       {src.count}")
        print(f"  Compression: {src.compression}")

        # Check a small window isn't all black
        window = rasterio.windows.Window(
            src.width // 2, src.height // 2, 256, 256
        )
        sample = src.read(1, window=window)
        mean_val = sample.mean()
        print(f"  Center tile mean: {mean_val:.1f} "
              f"({'OK' if mean_val > 10 else '⚠ possibly empty'})")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def process_year(year, extent, output_dir, workers, skip_existing):
    """Process a single year: compute grid, download, mosaic."""
    service_name = SERVICES.get(year)
    if not service_name:
        print(f"\n  ⚠ No service configured for year {year}")
        return None

    zoom = MAX_ZOOM[year]
    output_path = Path(output_dir) / f"edmonds_{year}_image.tif"

    if skip_existing and output_path.exists():
        size_gb = output_path.stat().st_size / 1e9
        if size_gb > 0.1:  # Skip if file seems complete (>100 MB)
            print(f"\n  ⏭ Skipping {year} — already exists ({size_gb:.2f} GB)")
            return {"year": year, "status": "skipped", "file_size_gb": round(size_gb, 2)}

    print(f"\n{'='*60}")
    print(f"  YEAR {year}")
    print(f"{'='*60}")

    grid = compute_tile_grid(extent, zoom)
    result = download_and_mosaic(year, service_name, grid, str(output_path), workers)
    validate_output(str(output_path), extent)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Download King County aerial imagery for Edmonds extent"
    )
    parser.add_argument(
        "--reference", required=True,
        help="Path to existing 2020 Edmonds image (defines extent)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Output directory for yearly GeoTIFFs"
    )
    parser.add_argument(
        "--years", nargs="+", type=int, default=[2023, 2021, 2019, 2017, 2015, 2013],
        help="Years to download (default: 2013-2023)"
    )
    parser.add_argument(
        "--workers", type=int, default=16,
        help="Parallel download threads (default: 16)"
    )
    parser.add_argument(
        "--skip_existing", action="store_true", default=True,
        help="Skip years that already have output files (default: True)"
    )
    parser.add_argument(
        "--no_skip", action="store_true",
        help="Re-download even if output exists"
    )

    args = parser.parse_args()
    if args.no_skip:
        args.skip_existing = False

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Get reference extent
    print("=" * 60)
    print("  REFERENCE IMAGE")
    print("=" * 60)
    extent = get_reference_extent(args.reference)

    # Process each year
    results = []
    for year in sorted(args.years):
        result = process_year(year, extent, args.output_dir, args.workers, args.skip_existing)
        if result:
            results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    total_gb = 0
    for r in results:
        status = r.get("status", "complete")
        gb = r.get("file_size_gb", 0)
        total_gb += gb
        print(f"  {r['year']}: {status} ({gb:.2f} GB)")
    print(f"\n  Total disk usage: {total_gb:.1f} GB")
    print(f"  Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
