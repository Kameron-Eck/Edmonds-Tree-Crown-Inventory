#!/usr/bin/env python3
"""
Edmonds Urban Tree Crown Detection Pipeline
============================================
Deep Distance Transform (DDT) approach for individual tree crown
detection from 7.5 cm aerial RGB imagery.

Author:  Kam (Edmonds Climate Advisory Board)
License: MIT

Pipeline stages:
    1. discover   — Auto-discover training sites from photos/polygons dirs
    2. inspect    — Raster + shapefile QA with band statistics
    3. preprocess — CRS harmonization, geometry repair, sliver removal
    4. dtm        — Per-crown centroid-based distance transform generation
    5. tile       — 512×512 paired RGB/DTM tiles with stratified split
    6. train      — U-Net (ResNet-101) with 5-fold CV, weighted sampling
    7. evaluate   — Watershed segmentation → IoU matching → F1 by size class
    8. infer      — Streaming full-image inference → GeoPackage crowns
    9. report     — Automated PDF report with figures and statistics

Usage:
    # Run full pipeline
    python scripts/pipeline.py --config configs/default.yaml

    # Run specific stages
    python scripts/pipeline.py --config configs/default.yaml --stages discover preprocess dtm tile train

    # Resume training from checkpoint
    python scripts/pipeline.py --config configs/default.yaml --stages train --resume
"""

import argparse
import gc
import io
import multiprocessing
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import albumentations as A
import fiona
import fiona.crs
import geopandas as gpd
import matplotlib
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.features
import rasterio.transform
import rasterio.windows
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import yaml
from albumentations.pytorch import ToTensorV2
from scipy.ndimage import label as scipy_label
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from shapely.validation import explain_validity
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

DEFAULT_CONFIG = {
    # -- Paths --
    "photos_dir": "data/photos",
    "polygons_dir": "data/polygons",
    "labels_dir": "data/labels",
    "tiles_dir": "data/tiles",
    "checkpoint_dir": "checkpoints",
    "inference_dir": "outputs/inference",
    "reports_dir": "outputs/reports",
    "boundary_path": "data/boundary/Edmonds_Boundary.shp",
    "hydro_path": None,
    "full_image_path": None,

    # -- CRS --
    "target_crs": "EPSG:3857",
    "expected_res_cm": 7.5,

    # -- Preprocessing --
    "small_area_threshold_m2": 0.5,

    # -- Tiling --
    "tile_size": 512,
    "stride": 512,
    "min_crown_frac": 0.00,
    "negative_sample_rate": 0.15,
    "test_frac": 0.20,
    "random_seed": 42,

    # -- Model --
    "encoder": "resnet101",
    "encoder_weights": "imagenet",
    "decoder_channels": [1024, 512, 256, 128, 64],
    "decoder_dropout": 0.3,

    # -- Training --
    "k_folds": 5,
    "epochs": 100,
    "batch_size": 30,
    "lr": 1e-4,
    "lr_patience": 8,
    "lr_factor": 0.5,
    "early_stop_patience": 25,
    "l1_lambda": 1e-6,
    "warmup_epochs": 5,
    "num_workers": 16,
    "save_every": 10,

    # -- Evaluation --
    "min_distance": 30,
    "dtm_threshold": 10.0,
    "min_crown_area_m2": 2.0,
    "iou_threshold": 0.5,
    "size_small_max_m2": 4.9,
    "size_medium_max_m2": 15.9,

    # -- Inference --
    "infer_stride": 256,
    "infer_batch_size": 160,
    "chunk_size": 2048,
    "chunk_border": 300,
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_config(config_path=None):
    """Load config from YAML file, falling back to defaults."""
    cfg = DEFAULT_CONFIG.copy()
    if config_path and Path(config_path).exists():
        with open(config_path) as f:
            user_cfg = yaml.safe_load(f)
        if user_cfg:
            cfg.update(user_cfg)
    return cfg


# ═══════════════════════════════════════════════════════════════
# STAGE 1: SITE DISCOVERY
# ═══════════════════════════════════════════════════════════════

def discover_sites(cfg):
    """Auto-discover training sites from photos directory.

    Naming convention:
        Photos:   <SiteName>_rgb.tif
        Polygons: <SiteName>.shp (omit for true negatives)
    """
    photos_dir = Path(cfg["photos_dir"])
    polygons_dir = Path(cfg["polygons_dir"])

    photo_files = sorted(photos_dir.glob("*_rgb.tif"))
    if not photo_files:
        raise FileNotFoundError(f"No *_rgb.tif files in {photos_dir}")

    image_paths, shapefile_paths, site_labels = [], [], []

    print("\n── Site Discovery ──")
    for photo_path in photo_files:
        label = photo_path.stem.replace("_rgb", "")
        shp_path = polygons_dir / f"{label}.shp"
        shp = str(shp_path) if shp_path.exists() else None

        image_paths.append(str(photo_path))
        shapefile_paths.append(shp)
        site_labels.append(label)

        status = f"✓ {shp_path.name}" if shp else "— true negative"
        print(f"  {label:<25} {status}")

    print(f"\n  Sites: {len(site_labels)}  |  "
          f"Positive: {sum(s is not None for s in shapefile_paths)}  |  "
          f"Negative: {sum(s is None for s in shapefile_paths)}")

    return image_paths, shapefile_paths, site_labels


# ═══════════════════════════════════════════════════════════════
# STAGE 2: INSPECTION
# ═══════════════════════════════════════════════════════════════

def inspect_raster(path, expected_res_cm=7.5):
    """Print raster properties and band statistics."""
    p = Path(path)
    with rasterio.open(path) as src:
        print(f"\n  {p.name}: {src.width}×{src.height} px | "
              f"CRS={src.crs} | Bands={src.count} | "
              f"Res={src.res[0]*100:.2f} cm")

        for i in range(1, src.count + 1):
            band = src.read(i).astype(np.float32)
            if src.nodata is not None:
                band = np.where(band == src.nodata, np.nan, band)
            print(f"    Band {i}: min={np.nanmin(band):.0f} "
                  f"mean={np.nanmean(band):.0f} max={np.nanmax(band):.0f}")


def inspect_shapefile(path):
    """Print shapefile properties and return GeoDataFrame."""
    gdf = gpd.read_file(path)
    gdf["_area_m2"] = gdf.geometry.area

    invalid = (~gdf.geometry.is_valid).sum()
    print(f"\n  {Path(path).name}: {len(gdf)} features | "
          f"CRS={gdf.crs} | Invalid={invalid}")
    print(f"    Area (m²): min={gdf['_area_m2'].min():.1f} "
          f"median={gdf['_area_m2'].median():.1f} "
          f"max={gdf['_area_m2'].max():.1f}")
    return gdf


def run_inspection(cfg, image_paths, shapefile_paths, site_labels):
    """Run QA inspection on all rasters and shapefiles."""
    print("\n── Raster Inspection ──")
    for path in image_paths:
        inspect_raster(path, cfg["expected_res_cm"])

    print("\n── Shapefile Inspection ──")
    raw_gdfs = {}
    for path, label in zip(shapefile_paths, site_labels):
        if path is not None:
            raw_gdfs[label] = inspect_shapefile(path)
    return raw_gdfs


# ═══════════════════════════════════════════════════════════════
# STAGE 3: PREPROCESSING & HARMONIZATION
# ═══════════════════════════════════════════════════════════════

def size_class(area_m2, cfg):
    """Classify crown by area."""
    if area_m2 <= cfg["size_small_max_m2"]:
        return "small"
    elif area_m2 <= cfg["size_medium_max_m2"]:
        return "medium"
    return "large"


def preprocess_site(label, gdf, cfg):
    """Clean and harmonize one site's GeoDataFrame."""
    target_crs = cfg["target_crs"]
    threshold = cfg["small_area_threshold_m2"]

    # Reproject
    if gdf.crs.to_epsg() != int(target_crs.split(":")[1]):
        gdf = gdf.to_crs(target_crs)

    # Explode multi-geometries
    if "MultiPolygon" in gdf.geometry.geom_type.unique():
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # Fix invalid geometries
    invalid_mask = ~gdf.geometry.is_valid
    if invalid_mask.sum() > 0:
        gdf["geometry"] = gdf.geometry.buffer(0)
        gdf = gdf[gdf.geometry.is_valid].reset_index(drop=True)

    # Second explode (buffer(0) can create MultiPolygons)
    if "MultiPolygon" in gdf.geometry.geom_type.unique():
        gdf = gdf.explode(index_parts=False).reset_index(drop=True)

    # Drop empties and slivers
    gdf = gdf[~gdf.geometry.is_empty].reset_index(drop=True)
    gdf["area_m2"] = gdf.geometry.area
    gdf = gdf[gdf["area_m2"] >= threshold].reset_index(drop=True)

    # Standardize columns
    gdf["area_m2"] = gdf.geometry.area
    gdf["crown_id"] = [f"{label}_{i}" for i in range(len(gdf))]
    gdf["site"] = label
    gdf["size_class"] = gdf["area_m2"].apply(lambda a: size_class(a, cfg))
    gdf = gdf[["crown_id", "site", "area_m2", "size_class", "geometry"]].copy()

    return gdf


def run_preprocessing(cfg, raw_gdfs, site_labels):
    """Preprocess all sites."""
    target_crs = cfg["target_crs"]
    gdfs = {}

    print("\n── Preprocessing ──")
    for label, gdf in raw_gdfs.items():
        gdfs[label] = preprocess_site(label, gdf, cfg)
        g = gdfs[label]
        print(f"  {label}: {len(g)} crowns | "
              f"S={sum(g['size_class']=='small')} "
              f"M={sum(g['size_class']=='medium')} "
              f"L={sum(g['size_class']=='large')}")

    # Fill true negatives
    for label in site_labels:
        if label not in gdfs:
            gdfs[label] = gpd.GeoDataFrame(
                columns=["crown_id", "site", "area_m2", "size_class", "geometry"],
                geometry="geometry",
            ).set_crs(target_crs)

    total = sum(len(g) for g in gdfs.values())
    print(f"\n  Total crowns: {total}")
    return gdfs


# ═══════════════════════════════════════════════════════════════
# STAGE 4: DISTANCE TRANSFORM GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_distance_transform(img_path, gdf, label, output_dir):
    """Generate centroid-based normalized distance transform for one site.

    For each crown polygon:
      1. Rasterize to pixel mask
      2. Compute Euclidean distance from each pixel to crown centroid
      3. Normalize: 100 at centroid → 1 at boundary → 0 outside

    Returns (output_path, stats_dict).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{label.lower()}_dtm.tif"

    with rasterio.open(img_path) as src:
        height, width = src.height, src.width
        transform = src.transform
        crs, res = src.crs, src.res[0]

    profile = {
        "driver": "GTiff", "dtype": "float32",
        "width": width, "height": height, "count": 1,
        "crs": crs, "transform": transform,
        "compress": "lzw", "nodata": 0.0,
    }

    # True negative → all-zero DTM
    if len(gdf) == 0:
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(np.zeros((height, width), dtype=np.float32), 1)
        return out_path, {"label": label, "crowns_total": 0,
                          "crowns_processed": 0, "crowns_skipped": 0}

    dtm = np.zeros((height, width), dtype=np.float32)
    gdf = gdf.copy().reset_index(drop=True)
    gdf["_int_id"] = gdf.index + 1

    # Rasterize all crowns
    shapes = ((geom, iid) for geom, iid in zip(gdf.geometry, gdf["_int_id"]))
    crown_mask = rasterio.features.rasterize(
        shapes, out_shape=(height, width), transform=transform,
        fill=0, dtype=np.int32, all_touched=False,
    )

    skipped, processed = 0, 0

    for _, row in tqdm(gdf.iterrows(), total=len(gdf),
                       desc=f"  DTM {label}", leave=False):
        int_id = int(row["_int_id"])
        crown_pixels = crown_mask == int_id
        if crown_pixels.sum() == 0:
            skipped += 1
            continue

        # Map centroid to pixel coordinates
        cx_map, cy_map = row.geometry.centroid.x, row.geometry.centroid.y
        cx_px = int(np.clip(np.round(
            (cx_map - transform.c) / transform.a), 0, width - 1))
        cy_px = int(np.clip(np.round(
            (cy_map - transform.f) / transform.e), 0, height - 1))

        # Crop to bounding box + 1px border
        rows_idx, cols_idx = np.where(crown_pixels)
        r_min = max(0, rows_idx.min() - 1)
        r_max = min(height - 1, rows_idx.max() + 1)
        c_min = max(0, cols_idx.min() - 1)
        c_max = min(width - 1, cols_idx.max() + 1)

        crop = crown_pixels[r_min:r_max + 1, c_min:c_max + 1]
        cy_rel = np.clip(cy_px - r_min, 0, crop.shape[0] - 1)
        cx_rel = np.clip(cx_px - c_min, 0, crop.shape[1] - 1)

        rc, cc = np.where(crop)
        dist = np.sqrt((rc - cy_rel) ** 2 + (cc - cx_rel) ** 2)
        d_max = dist.max()

        if d_max == 0:
            norm = np.ones_like(dist, dtype=np.float32) * 100.0
        else:
            norm = (1.0 - dist / d_max) * 99.0 + 1.0

        dtm[rc + r_min, cc + c_min] = norm.astype(np.float32)
        processed += 1

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(dtm, 1)

    crown_pix = (dtm > 0).sum()
    stats = {
        "label": label, "crowns_total": len(gdf),
        "crowns_processed": processed, "crowns_skipped": skipped,
        "crown_coverage_pct": 100 * crown_pix / (height * width),
        "dtm_min": float(dtm[dtm > 0].min()) if crown_pix > 0 else 0.0,
        "dtm_max": float(dtm.max()),
    }
    return out_path, stats


def run_dtm_generation(cfg, gdfs, image_paths, site_labels):
    """Generate distance transforms for all sites."""
    output_dir = Path(cfg["labels_dir"])
    image_paths_dict = dict(zip(site_labels, image_paths))

    print("\n── Distance Transform Generation ──")
    dtm_paths, all_stats = {}, {}

    for label in site_labels:
        out_path, stats = generate_distance_transform(
            image_paths_dict[label], gdfs[label], label, output_dir)
        dtm_paths[label] = out_path
        all_stats[label] = stats
        print(f"  {label}: {stats['crowns_processed']}/{stats['crowns_total']} crowns | "
              f"skipped={stats['crowns_skipped']}")

    return dtm_paths, all_stats


# ═══════════════════════════════════════════════════════════════
# STAGE 5: TILING
# ═══════════════════════════════════════════════════════════════

def tile_site(label, img_path, dtm_path, cfg):
    """Slice one site into tile_size × tile_size patches."""
    tile_size = cfg["tile_size"]
    stride = cfg["stride"]
    neg_rate = cfg["negative_sample_rate"]
    records = []

    with rasterio.open(img_path) as img_src, \
         rasterio.open(dtm_path) as dtm_src:

        height, width = img_src.height, img_src.width
        assert (height, width) == (dtm_src.height, dtm_src.width)
        img_transform, crs = img_src.transform, img_src.crs

        for row_off in range(0, height - tile_size + 1, stride):
            for col_off in range(0, width - tile_size + 1, stride):
                window = rasterio.windows.Window(
                    col_off=col_off, row_off=row_off,
                    width=tile_size, height=tile_size)

                img_tile = img_src.read(window=window)
                dtm_tile = dtm_src.read(1, window=window)
                crown_frac = float((dtm_tile > 0).sum()) / (tile_size ** 2)

                if crown_frac < cfg["min_crown_frac"]:
                    if crown_frac == 0.0 and np.random.random() < neg_rate:
                        pass  # keep as true negative
                    else:
                        continue

                tile_tf = rasterio.windows.transform(window, img_transform)
                dtm_vals = dtm_tile[dtm_tile > 0]

                records.append({
                    "tile_name": f"{label.lower()}_r{row_off:05d}_c{col_off:05d}.tif",
                    "site": label, "row_off": row_off, "col_off": col_off,
                    "crown_frac": round(float(crown_frac), 4),
                    "dtm_mean": round(float(dtm_vals.mean()), 2) if len(dtm_vals) else 0.0,
                    "dtm_max": round(float(dtm_vals.max()), 2) if len(dtm_vals) else 0.0,
                    "tile_transform": tile_tf, "crs": crs,
                    "_img_tile": img_tile, "_dtm_tile": dtm_tile,
                })
    return records


def run_tiling(cfg, dtm_paths, image_paths, site_labels):
    """Tile all sites and write to disk with stratified train/test split."""
    tile_dir = Path(cfg["tiles_dir"])
    tile_size = cfg["tile_size"]

    for split in ["train", "test"]:
        (tile_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (tile_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    image_paths_dict = dict(zip(site_labels, image_paths))

    print("\n── Tiling ──")
    all_records = []
    for label in site_labels:
        records = tile_site(
            label, image_paths_dict[label], str(dtm_paths[label]), cfg)
        all_records.extend(records)
        print(f"  {label}: {len(records)} tiles accepted")

    print(f"  Total: {len(all_records)} tiles")

    # Stratified split
    tile_names = [r["tile_name"] for r in all_records]
    sites = [r["site"] for r in all_records]
    train_names, test_names = train_test_split(
        tile_names, test_size=cfg["test_frac"],
        stratify=sites, random_state=cfg["random_seed"])

    train_set, test_set = set(train_names), set(test_names)

    # Write tiles
    img_profile = {"driver": "GTiff", "dtype": "uint8", "count": 3,
                   "width": tile_size, "height": tile_size, "compress": "lzw"}
    dtm_profile = {"driver": "GTiff", "dtype": "float32", "count": 1,
                   "width": tile_size, "height": tile_size,
                   "compress": "lzw", "nodata": 0.0}

    index_rows = []
    for rec in tqdm(all_records, desc="  Writing tiles"):
        split = "train" if rec["tile_name"] in train_set else "test"
        img_out = tile_dir / split / "images" / rec["tile_name"]
        dtm_out = tile_dir / split / "labels" / rec["tile_name"]

        with rasterio.open(img_out, "w", **{**img_profile,
                           "crs": rec["crs"],
                           "transform": rec["tile_transform"]}) as dst:
            dst.write(rec["_img_tile"])

        with rasterio.open(dtm_out, "w", **{**dtm_profile,
                           "crs": rec["crs"],
                           "transform": rec["tile_transform"]}) as dst:
            dst.write(rec["_dtm_tile"], 1)

        index_rows.append({
            "tile_name": rec["tile_name"], "site": rec["site"],
            "split": split, "row_off": rec["row_off"],
            "col_off": rec["col_off"], "crown_frac": rec["crown_frac"],
            "dtm_mean": rec["dtm_mean"], "dtm_max": rec["dtm_max"],
            "img_path": str(img_out), "dtm_path": str(dtm_out),
        })

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(tile_dir / "tile_index.csv", index=False)

    print(f"  Train: {len(train_names)} | Test: {len(test_names)}")
    return index_df


# ═══════════════════════════════════════════════════════════════
# STAGE 6: MODEL + TRAINING
# ═══════════════════════════════════════════════════════════════

def make_spatial_transform():
    return A.Compose([
        A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5),
        A.Transpose(p=0.5), A.RandomRotate90(p=0.5),
        A.Rotate(limit=45, border_mode=0, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1,
                           rotate_limit=0, border_mode=0, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.4),
        A.ElasticTransform(alpha=50, sigma=5, p=0.3),
    ], additional_targets={"label": "mask"})


def make_pixel_transform():
    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.MedianBlur(blur_limit=5, p=1.0),
        ], p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30,
                             val_shift_limit=15, p=0.5),
        A.RandomGamma(gamma_limit=(70, 130), p=0.4),
        A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_limit=(1, 3),
                       shadow_dimension=5, p=0.4),
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, p=0.3),
        A.Downscale(scale_range=(0.5, 0.75),
                    interpolation_pair={"downscale": 0, "upscale": 2}, p=0.3),
        A.CoarseDropout(num_holes_range=(2, 8), hole_height_range=(32, 96),
                        hole_width_range=(32, 96), fill=0, p=0.4),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ])


def make_test_transform():
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(),
    ], additional_targets={"label": "mask"})


class TreeCrownDataset(Dataset):
    """Paired RGB + DTM tile dataset with augmentation."""

    def __init__(self, df, training=True):
        self.df = df.reset_index(drop=True)
        self.training = training
        if training:
            self.spatial_tf = make_spatial_transform()
            self.pixel_tf = make_pixel_transform()
        else:
            self.test_tf = make_test_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        with rasterio.open(row["img_path"]) as src:
            img = src.read().transpose(1, 2, 0)
        with rasterio.open(row["dtm_path"]) as src:
            lbl = src.read(1).astype(np.float32)

        if self.training:
            out = self.spatial_tf(image=img, label=lbl)
            img, lbl = out["image"], out["label"]
            out = self.pixel_tf(image=img)
            img = out["image"]
            lbl = torch.from_numpy(lbl).unsqueeze(0)
        else:
            out = self.test_tf(image=img, label=lbl)
            img, lbl = out["image"], out["label"]
            if lbl.dim() == 2:
                lbl = lbl.unsqueeze(0)

        meta = {"tile_name": row["tile_name"], "site": row["site"],
                "crown_frac": float(row["crown_frac"])}
        return img, lbl, meta


def _inject_dropout(module, p):
    """Inject Dropout2d into decoder Sequential blocks."""
    for name, child in module.named_children():
        if isinstance(child, nn.Sequential):
            child.add_module("dropout", nn.Dropout2d(p=p))
        else:
            _inject_dropout(child, p)


def build_model(cfg, device):
    """Build U-Net with ResNet encoder and decoder dropout."""
    m = smp.Unet(
        encoder_name=cfg["encoder"],
        encoder_weights=cfg["encoder_weights"],
        decoder_channels=tuple(cfg["decoder_channels"]),
        in_channels=3, classes=1, activation=None,
    )
    _inject_dropout(m.decoder, cfg["decoder_dropout"])
    m = m.to(device)
    m = torch.compile(m)
    return m


def train_one_epoch(model, loader, optimizer, scaler, criterion, l1_lambda, device):
    model.train()
    total_sum, mae_sum, n = 0.0, 0.0, 0
    for imgs, lbls, _ in loader:
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            preds = model(imgs)
            mae = criterion(preds, lbls)
            l1_pen = sum(p.abs().sum() for p in model.parameters())
            loss = mae + l1_lambda * l1_pen
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        total_sum += loss.item()
        mae_sum += mae.item()
        n += 1
    return total_sum / n, mae_sum / n


def validate(model, loader, criterion, device):
    model.eval()
    mae_sum, n = 0.0, 0
    with torch.no_grad():
        for imgs, lbls, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            lbls = lbls.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                preds = model(imgs)
                mae = criterion(preds, lbls)
            mae_sum += mae.item()
            n += 1
    return mae_sum / n


def save_checkpoint(fold, epoch, model, optimizer, sched, history, best_val, path):
    model_state = (model._orig_mod.state_dict()
                   if hasattr(model, "_orig_mod") else model.state_dict())
    torch.save({
        "fold": fold, "epoch": epoch, "model_state": model_state,
        "optim_state": optimizer.state_dict(),
        "sched_state": sched.state_dict(),
        "history": history, "best_val": best_val,
    }, path)


def run_training(cfg, resume=False):
    """5-fold cross-validation training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_dir = Path(cfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    index_df = pd.read_csv(Path(cfg["tiles_dir"]) / "tile_index.csv")
    criterion = nn.L1Loss()
    kf = KFold(n_splits=cfg["k_folds"], shuffle=True, random_state=42)

    all_fold_histories = []
    global_best_val = float("inf")
    global_best_ckpt = ckpt_dir / "ddt_best_global.pt"

    print(f"\n── Training: {cfg['k_folds']}-Fold CV | {cfg['epochs']} epochs/fold ──")
    print(f"  Device: {device}")

    for fold, (train_idx, val_idx) in enumerate(kf.split(index_df)):
        fold_num = fold + 1
        fold_train = index_df.iloc[train_idx].reset_index(drop=True)
        fold_val = index_df.iloc[val_idx].reset_index(drop=True)

        # Weighted sampler
        site_counts = fold_train["site"].value_counts().to_dict()
        weights = fold_train["site"].map(lambda s: 1.0 / site_counts[s]).values
        sampler = WeightedRandomSampler(
            torch.from_numpy(weights.astype(np.float32)),
            num_samples=len(fold_train), replacement=True)

        train_loader = DataLoader(
            TreeCrownDataset(fold_train, training=True),
            batch_size=cfg["batch_size"], sampler=sampler,
            num_workers=cfg["num_workers"], pin_memory=True,
            drop_last=True, persistent_workers=True, prefetch_factor=4)
        val_loader = DataLoader(
            TreeCrownDataset(fold_val, training=False),
            batch_size=cfg["batch_size"], shuffle=False,
            num_workers=cfg["num_workers"], pin_memory=True,
            persistent_workers=True, prefetch_factor=4)

        model = build_model(cfg, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
        warmup = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda e: (e + 1) / cfg["warmup_epochs"]
            if e < cfg["warmup_epochs"] else 1.0)
        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg["lr_factor"],
            patience=cfg["lr_patience"], min_lr=1e-6)
        scaler = torch.cuda.amp.GradScaler()

        best_val = float("inf")
        early_count = 0
        history = {"train_total": [], "train_mae": [], "val_mae": []}

        print(f"\n  Fold {fold_num}: train={len(fold_train)} val={len(fold_val)}")

        for epoch in range(cfg["epochs"]):
            t0 = time.time()
            train_total, train_mae = train_one_epoch(
                model, train_loader, optimizer, scaler,
                criterion, cfg["l1_lambda"], device)
            val_mae = validate(model, val_loader, criterion, device)

            if epoch < cfg["warmup_epochs"]:
                warmup.step()
            else:
                plateau.step(val_mae)

            history["train_total"].append(train_total)
            history["train_mae"].append(train_mae)
            history["val_mae"].append(val_mae)

            is_best = val_mae < best_val
            marker = ""
            if is_best:
                best_val = val_mae
                early_count = 0
                save_checkpoint(fold_num, epoch, model, optimizer,
                                plateau, history, best_val,
                                ckpt_dir / f"ddt_best_fold{fold_num}.pt")
                if best_val < global_best_val:
                    global_best_val = best_val
                    save_checkpoint(fold_num, epoch, model, optimizer,
                                    plateau, history, best_val,
                                    global_best_ckpt)
                    marker = " ◆GLOBAL"
                else:
                    marker = " ★"
            else:
                early_count += 1

            lr = optimizer.param_groups[0]["lr"]
            print(f"    E{epoch+1:>3} train={train_mae:.4f} "
                  f"val={val_mae:.4f} lr={lr:.1e} "
                  f"t={time.time()-t0:.0f}s{marker}")

            if early_count >= cfg["early_stop_patience"]:
                print(f"    Early stop at epoch {epoch+1}")
                break

        all_fold_histories.append({
            "fold": fold_num, "best_val": best_val, **history})

        del model, optimizer, scaler
        torch.cuda.empty_cache()

    # Summary
    best_vals = [h["best_val"] for h in all_fold_histories]
    print(f"\n  CV Summary: {np.mean(best_vals):.4f} ± {np.std(best_vals):.4f}")
    print(f"  Global best: {global_best_val:.4f}")

    # Save loss history
    rows = []
    for h in all_fold_histories:
        for i, (tt, tm, vm) in enumerate(
                zip(h["train_total"], h["train_mae"], h["val_mae"]), 1):
            rows.append({"fold": h["fold"], "epoch": i,
                         "train_total": tt, "train_mae": tm, "val_mae": vm})
    pd.DataFrame(rows).to_csv(ckpt_dir / "loss_history.csv", index=False)

    return all_fold_histories


# ═══════════════════════════════════════════════════════════════
# STAGE 7: EVALUATION
# ═══════════════════════════════════════════════════════════════

def dtm_to_polygons(dtm, pixel_area_m2, min_distance, dtm_threshold,
                    min_crown_area_m2):
    """Watershed segmentation: DTM → list of crown polygon dicts."""
    mask = dtm >= dtm_threshold
    if mask.sum() == 0:
        return []

    coords = peak_local_max(dtm, min_distance=min_distance,
                            labels=mask, threshold_abs=dtm_threshold)
    if len(coords) == 0:
        return []

    seeds = np.zeros(dtm.shape, dtype=bool)
    seeds[tuple(coords.T)] = True
    markers, _ = scipy_label(seeds)
    ws = watershed(-dtm, markers, mask=mask)

    polys = []
    for cid in np.unique(ws):
        if cid == 0:
            continue
        px = ws == cid
        area = px.sum() * pixel_area_m2
        if area < min_crown_area_m2:
            continue
        rows, cols = np.where(px)
        polys.append({
            "row_min": int(rows.min()), "row_max": int(rows.max()),
            "col_min": int(cols.min()), "col_max": int(cols.max()),
            "area_m2": float(area), "mask": px,
        })
    return polys


def mask_iou(a, b):
    inter = (a & b).sum()
    union = (a | b).sum()
    return inter / union if union > 0 else 0.0


def evaluate_tile(pred_polys, gt_polys, iou_thresh):
    """Greedy IoU matching → TP/FP/FN lists."""
    matched = set()
    TPs, FPs, FNs = [], [], []

    for gt in gt_polys:
        best_iou, best_idx = 0.0, -1
        for pi, pred in enumerate(pred_polys):
            if pi in matched:
                continue
            if (pred["row_max"] < gt["row_min"] or pred["row_min"] > gt["row_max"] or
                    pred["col_max"] < gt["col_min"] or pred["col_min"] > gt["col_max"]):
                continue
            iou = mask_iou(pred["mask"], gt["mask"])
            if iou > best_iou:
                best_iou, best_idx = iou, pi

        if best_iou >= iou_thresh:
            TPs.append({"gt_area": gt["area_m2"],
                        "pred_area": pred_polys[best_idx]["area_m2"], "iou": best_iou})
            matched.add(best_idx)
        else:
            FNs.append({"gt_area": gt["area_m2"]})

    for pi, pred in enumerate(pred_polys):
        if pi not in matched:
            FPs.append({"pred_area": pred["area_m2"]})

    return TPs, FPs, FNs


def compute_f1(TPs, FPs, FNs):
    tp, fp, fn = len(TPs), len(FPs), len(FNs)
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, tp, fp, fn


def run_evaluation(cfg):
    """Evaluate best model on test tiles."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tile_dir = Path(cfg["tiles_dir"])
    ckpt_dir = Path(cfg["checkpoint_dir"])

    index_df = pd.read_csv(tile_dir / "tile_index.csv")
    test_df = index_df[index_df["split"] == "test"].reset_index(drop=True)

    # Derive pixel area from tile transform
    with rasterio.open(test_df.iloc[0]["img_path"]) as src:
        px_area = abs(src.transform.a * src.transform.e)

    # Load model
    model = smp.Unet(
        encoder_name=cfg["encoder"], encoder_weights=None,
        decoder_channels=tuple(cfg["decoder_channels"]),
        in_channels=3, classes=1, activation=None).to(device)
    ckpt = torch.load(ckpt_dir / "ddt_best_global.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    eval_tf = A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2()])

    # Inference
    print(f"\n── Evaluation on {len(test_df)} test tiles ──")
    predictions, ground_truths = {}, {}

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="  Inference"):
            with rasterio.open(row["img_path"]) as src:
                img = src.read().transpose(1, 2, 0)
            with rasterio.open(row["dtm_path"]) as src:
                gt = src.read(1).astype(np.float32)

            inp = eval_tf(image=img)["image"].unsqueeze(0).to(device)
            pred = model(inp).squeeze().cpu().numpy()
            predictions[row["tile_name"]] = np.clip(pred, 0, 100)
            ground_truths[row["tile_name"]] = gt

    # Watershed + matching
    sm, mm = cfg["size_small_max_m2"], cfg["size_medium_max_m2"]

    def sc(a):
        return "small" if a <= sm else ("medium" if a <= mm else "large")

    all_tp = {"overall": [], "small": [], "medium": [], "large": []}
    all_fp = {"overall": [], "small": [], "medium": [], "large": []}
    all_fn = {"overall": [], "small": [], "medium": [], "large": []}
    site_results = {}

    for _, row in test_df.iterrows():
        tn = row["tile_name"]
        pp = dtm_to_polygons(predictions[tn], px_area,
                             cfg["min_distance"], cfg["dtm_threshold"],
                             cfg["min_crown_area_m2"])
        gp = dtm_to_polygons(ground_truths[tn], px_area,
                             cfg["min_distance"], cfg["dtm_threshold"],
                             cfg["min_crown_area_m2"])
        TPs, FPs, FNs = evaluate_tile(pp, gp, cfg["iou_threshold"])

        all_tp["overall"].extend(TPs)
        all_fp["overall"].extend(FPs)
        all_fn["overall"].extend(FNs)

        for tp in TPs:
            all_tp[sc(tp["gt_area"])].append(tp)
        for fn in FNs:
            all_fn[sc(fn["gt_area"])].append(fn)
        for fp in FPs:
            all_fp[sc(fp["pred_area"])].append(fp)

        site = row["site"]
        if site not in site_results:
            site_results[site] = {"TPs": [], "FPs": [], "FNs": []}
        site_results[site]["TPs"].extend(TPs)
        site_results[site]["FPs"].extend(FPs)
        site_results[site]["FNs"].extend(FNs)

    # Print results
    print(f"\n  {'Class':<10} {'Prec':>7} {'Rec':>7} {'F1':>7} "
          f"{'TP':>5} {'FP':>5} {'FN':>5}")
    for cls in ["small", "medium", "large", "overall"]:
        p, r, f1, tp, fp, fn = compute_f1(all_tp[cls], all_fp[cls], all_fn[cls])
        lbl = cls.upper() if cls == "overall" else cls.capitalize()
        print(f"  {lbl:<10} {p:>7.3f} {r:>7.3f} {f1:>7.3f} {tp:>5} {fp:>5} {fn:>5}")

    # Save results
    eval_rows = []
    for cls in ["small", "medium", "large", "overall"]:
        p, r, f1, tp, fp, fn = compute_f1(all_tp[cls], all_fp[cls], all_fn[cls])
        eval_rows.append({"grouping": "size_class", "label": cls,
                          "precision": round(p, 4), "recall": round(r, 4),
                          "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn})
    for site in sorted(site_results):
        p, r, f1, tp, fp, fn = compute_f1(
            site_results[site]["TPs"], site_results[site]["FPs"],
            site_results[site]["FNs"])
        eval_rows.append({"grouping": "site", "label": site,
                          "precision": round(p, 4), "recall": round(r, 4),
                          "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn})

    pd.DataFrame(eval_rows).to_csv(ckpt_dir / "eval_results.csv", index=False)
    print(f"\n  Results saved: {ckpt_dir / 'eval_results.csv'}")
    return eval_rows


# ═══════════════════════════════════════════════════════════════
# STAGE 8: FULL-IMAGE INFERENCE
# ═══════════════════════════════════════════════════════════════

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


def run_inference(cfg):
    """Stream inference over full Edmonds image → DTM → GeoPackage."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    full_img = Path(cfg["full_image_path"])
    ckpt_path = Path(cfg["checkpoint_dir"]) / "ddt_best_global.pt"
    out_dir = Path(cfg["inference_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    dtm_out = out_dir / "edmonds_dtm.tif"
    crowns_out = out_dir / "edmonds_crowns.gpkg"
    stats_out = out_dir / "edmonds_crown_stats.csv"

    tile_size = cfg["tile_size"]
    stride = cfg["infer_stride"]
    center_crop = stride
    pad = (tile_size - stride) // 2
    batch_size = cfg["infer_batch_size"]

    # Image metadata
    with rasterio.open(full_img) as src:
        h, w = src.height, src.width
        crs, tf, bounds = src.crs, src.transform, src.bounds
        px_x, px_y = tf.a, abs(tf.e)
        px_area = px_x * px_y

    print(f"\n── Inference: {w}×{h} px ──")

    # Load model
    model = smp.Unet(
        encoder_name=cfg["encoder"], encoder_weights=None,
        decoder_channels=tuple(cfg["decoder_channels"]),
        in_channels=3, classes=1, activation=None).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

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

    # Streaming inference
    dtm_profile = {
        "driver": "GTiff", "dtype": "float32", "width": w, "height": h,
        "count": 1, "crs": crs, "transform": tf, "compress": "lzw",
        "nodata": 0.0, "BIGTIFF": "YES",
    }

    batch_imgs, batch_coords = [], []

    def flush(imgs, coords, dst):
        if not imgs:
            return
        inp = torch.stack(imgs).to(device)
        with torch.no_grad():
            pred = model(inp).squeeze(1).cpu().numpy()
        pred = np.clip(pred, 0, 100)
        for k, (ro, co) in enumerate(coords):
            c = pred[k, pad:pad + center_crop, pad:pad + center_crop]
            ch = min(ro + center_crop, h) - ro
            cw = min(co + center_crop, w) - co
            win = rasterio.windows.Window(co, ro, cw, ch)
            dst.write(c[:ch, :cw][np.newaxis].astype(np.float32), window=win)

    with rasterio.open(dtm_out, "w", **dtm_profile) as dst:
        with rasterio.open(full_img) as src:
            for ro, co in tqdm(origins, desc="  Inference", miniters=3000):
                rs, cs = ro - pad, co - pad
                re, ce = rs + tile_size, cs + tile_size

                rr0, rc0 = max(0, rs), max(0, cs)
                rr1, rc1 = min(h, re), min(w, ce)
                win = rasterio.windows.Window(rc0, rr0, rc1 - rc0, rr1 - rr0)
                tile = src.read([1, 2, 3], window=win).transpose(1, 2, 0)

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

    print(f"  DTM: {dtm_out} ({dtm_out.stat().st_size / 1e6:.0f} MB)")

    # Watershed
    print(f"\n  Watershed → {crowns_out.name}")
    n_workers = min(16, multiprocessing.cpu_count())
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
                            desc="  Watershed", mininterval=30):
                for rec in fut.result():
                    gpkg.writerecords([{
                        "geometry": rec["geometry"],
                        "properties": {
                            "crown_id": f"EDM_{crown_n:07d}",
                            "area_m2": rec["area_m2"],
                            "diameter_m": rec["diameter_m"],
                            "size_class": rec["size_class"],
                        }}])
                    crown_n += 1

    print(f"  Crowns: {crown_n:,} ({crowns_out.stat().st_size / 1e6:.0f} MB)")

    # Stats
    gdf = gpd.read_file(crowns_out)
    total_ha = gdf["area_m2"].sum() / 10_000
    cover_pct = 100 * gdf["area_m2"].sum() / (w * h * px_area)
    print(f"  Canopy: {total_ha:.1f} ha | Cover: {cover_pct:.1f}%")

    pd.DataFrame([
        {"metric": "total_crowns", "value": len(gdf)},
        {"metric": "total_canopy_ha", "value": round(total_ha, 2)},
        {"metric": "canopy_cover_pct", "value": round(cover_pct, 2)},
        {"metric": "median_area_m2", "value": round(gdf["area_m2"].median(), 2)},
        {"metric": "median_diameter_m", "value": round(gdf["diameter_m"].median(), 2)},
    ]).to_csv(stats_out, index=False)


# ═══════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ═══════════════════════════════════════════════════════════════

ALL_STAGES = ["discover", "inspect", "preprocess", "dtm", "tile",
              "train", "evaluate", "infer"]


def main():
    parser = argparse.ArgumentParser(
        description="Edmonds Urban Tree Crown Detection Pipeline")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--stages", nargs="+", default=ALL_STAGES,
                        choices=ALL_STAGES,
                        help="Pipeline stages to run")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from checkpoint")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stages = args.stages

    print("=" * 60)
    print("  Edmonds Urban Tree Crown Detection Pipeline")
    print(f"  Stages: {', '.join(stages)}")
    print("=" * 60)

    # Shared state passed between stages
    image_paths = shapefile_paths = site_labels = None
    raw_gdfs = gdfs = dtm_paths = None

    if "discover" in stages:
        image_paths, shapefile_paths, site_labels = discover_sites(cfg)

    if "inspect" in stages:
        assert image_paths is not None, "Run 'discover' first"
        raw_gdfs = run_inspection(cfg, image_paths, shapefile_paths, site_labels)

    if "preprocess" in stages:
        if raw_gdfs is None:
            # Re-discover and load
            image_paths, shapefile_paths, site_labels = discover_sites(cfg)
            raw_gdfs = {}
            for path, label in zip(shapefile_paths, site_labels):
                if path is not None:
                    raw_gdfs[label] = gpd.read_file(path)
        gdfs = run_preprocessing(cfg, raw_gdfs, site_labels)

    if "dtm" in stages:
        assert gdfs is not None and image_paths is not None
        dtm_paths, _ = run_dtm_generation(cfg, gdfs, image_paths, site_labels)

    if "tile" in stages:
        if dtm_paths is None:
            # Reconstruct paths from labels dir
            labels_dir = Path(cfg["labels_dir"])
            image_paths, shapefile_paths, site_labels = discover_sites(cfg)
            dtm_paths = {l: labels_dir / f"{l.lower()}_dtm.tif" for l in site_labels}
        run_tiling(cfg, dtm_paths, image_paths, site_labels)

    if "train" in stages:
        run_training(cfg, resume=args.resume)

    if "evaluate" in stages:
        run_evaluation(cfg)

    if "infer" in stages:
        assert cfg["full_image_path"], "Set full_image_path in config"
        run_inference(cfg)

    print("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
