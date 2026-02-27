#!/usr/bin/env python3
"""
Gather Repository Statistics for README
========================================
Reads file headers, CSVs, and GeoPackage metadata to collect
every number referenced in the README and docs.

Run in Colab after mounting Drive:
    !python scripts/gather_stats.py --config configs/colab.yaml

Outputs a YAML summary to stdout and saves to outputs/repo_stats.yaml
"""

import argparse
import json
import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def gather_stats(cfg):
    stats = {}

    # ══════════════════════════════════════════════════════════
    # 1. FULL CITY IMAGE
    # ══════════════════════════════════════════════════════════
    section("FULL CITY IMAGE")

    full_img = cfg.get("full_image_path")
    if full_img and Path(full_img).exists():
        with rasterio.open(full_img) as src:
            w, h = src.width, src.height
            px_x = src.res[0]
            px_y = abs(src.res[1])
            px_area = px_x * px_y
            bands = src.count
            dtype = src.dtypes[0]
            crs = str(src.crs)
            compress = src.compression

            extent_km_x = w * px_x / 1000
            extent_km_y = h * px_y / 1000
            total_pixels = w * h
            file_size_gb = Path(full_img).stat().st_size / 1e9

            # Uncompressed sizes
            bytes_per_px = {"uint8": 1, "uint16": 2, "float32": 4, "float64": 8}
            bpp = bytes_per_px.get(dtype, 1)
            uncompressed_gb = total_pixels * bands * bpp / 1e9
            float32_gb = total_pixels * bands * 4 / 1e9
            compression_ratio = uncompressed_gb / file_size_gb if file_size_gb > 0 else 0

        img_stats = {
            "width_px": w,
            "height_px": h,
            "total_pixels": total_pixels,
            "total_pixel_values": total_pixels * bands,
            "bands": bands,
            "dtype": dtype,
            "crs": crs,
            "compression": str(compress),
            "pixel_size_x_m": round(px_x, 6),
            "pixel_size_y_m": round(px_y, 6),
            "pixel_size_cm": round(px_x * 100, 2),
            "pixel_area_m2": round(px_area, 8),
            "extent_km_x": round(extent_km_x, 2),
            "extent_km_y": round(extent_km_y, 2),
            "extent_km_str": f"{extent_km_x:.1f} × {extent_km_y:.1f} km",
            "file_size_gb": round(file_size_gb, 2),
            "uncompressed_gb": round(uncompressed_gb, 2),
            "float32_memory_gb": round(float32_gb, 2),
            "compression_ratio": f"{compression_ratio:.1f}x",
        }
        stats["full_image"] = img_stats

        print(f"  File:              {Path(full_img).name}")
        print(f"  Dimensions:        {w:,} × {h:,} px")
        print(f"  Total pixels:      {total_pixels:,}")
        print(f"  Bands:             {bands} ({dtype})")
        print(f"  CRS:               {crs}")
        print(f"  Pixel size:        {px_x*100:.2f} cm × {px_y*100:.2f} cm")
        print(f"  Ground extent:     {extent_km_x:.2f} × {extent_km_y:.2f} km")
        print(f"  File size (disk):  {file_size_gb:.2f} GB")
        print(f"  Uncompressed:      {uncompressed_gb:.2f} GB")
        print(f"  As float32:        {float32_gb:.2f} GB")
        print(f"  Compression:       {compress} ({compression_ratio:.1f}x ratio)")
    else:
        print(f"  ⚠ Full image not found: {full_img}")
        stats["full_image"] = None

    # ══════════════════════════════════════════════════════════
    # 2. TRAINING SITE IMAGERY
    # ══════════════════════════════════════════════════════════
    section("TRAINING SITES")

    photos_dir = Path(cfg.get("photos_dir", "data/photos"))
    polygons_dir = Path(cfg.get("polygons_dir", "data/polygons"))
    site_stats = []

    if photos_dir.exists():
        for tif in sorted(photos_dir.glob("*_rgb.tif")):
            label = tif.stem.replace("_rgb", "")
            shp = polygons_dir / f"{label}.shp"

            with rasterio.open(tif) as src:
                s = {
                    "site": label,
                    "width_px": src.width,
                    "height_px": src.height,
                    "total_pixels": src.width * src.height,
                    "pixel_size_cm": round(src.res[0] * 100, 2),
                    "crs": str(src.crs),
                    "file_size_mb": round(tif.stat().st_size / 1e6, 1),
                    "has_annotations": shp.exists(),
                }

            if shp.exists():
                gdf = gpd.read_file(shp)
                s["crown_count"] = len(gdf)
                s["crown_area_min_m2"] = round(gdf.geometry.area.min(), 2)
                s["crown_area_median_m2"] = round(gdf.geometry.area.median(), 2)
                s["crown_area_max_m2"] = round(gdf.geometry.area.max(), 2)
                s["crown_area_mean_m2"] = round(gdf.geometry.area.mean(), 2)
            else:
                s["crown_count"] = 0

            site_stats.append(s)

            ann = f"{s['crown_count']} crowns" if s["has_annotations"] else "true negative"
            print(f"  {label:<25} {s['width_px']:>6}×{s['height_px']:<6} px | "
                  f"{s['file_size_mb']:>6.1f} MB | {ann}")

        stats["training_sites"] = site_stats
        stats["training_sites_summary"] = {
            "total_sites": len(site_stats),
            "positive_sites": sum(1 for s in site_stats if s["has_annotations"]),
            "negative_sites": sum(1 for s in site_stats if not s["has_annotations"]),
            "total_crowns": sum(s["crown_count"] for s in site_stats),
        }
        print(f"\n  Total sites: {len(site_stats)} | "
              f"Crowns: {stats['training_sites_summary']['total_crowns']}")
    else:
        print(f"  ⚠ Photos dir not found: {photos_dir}")

    # ══════════════════════════════════════════════════════════
    # 3. TILE INDEX
    # ══════════════════════════════════════════════════════════
    section("TILES")

    tile_index = Path(cfg.get("tiles_dir", "data/tiles")) / "tile_index.csv"
    if tile_index.exists():
        df = pd.read_csv(tile_index)
        train_df = df[df["split"] == "train"]
        test_df = df[df["split"] == "test"]

        tile_stats = {
            "total_tiles": len(df),
            "train_tiles": len(train_df),
            "test_tiles": len(test_df),
            "sites_in_tiles": sorted(df["site"].unique().tolist()),
            "crown_frac_mean": round(df["crown_frac"].mean(), 4),
            "crown_frac_median": round(df["crown_frac"].median(), 4),
            "tiles_per_site": df.groupby("site").size().to_dict(),
            "tiles_per_site_per_split": df.groupby(["site", "split"]).size().to_dict(),
        }
        stats["tiles"] = tile_stats

        print(f"  Total:  {len(df)} tiles")
        print(f"  Train:  {len(train_df)} | Test: {len(test_df)}")
        print(f"  Crown coverage: mean={df['crown_frac'].mean()*100:.1f}% "
              f"median={df['crown_frac'].median()*100:.1f}%")
        for site in sorted(df["site"].unique()):
            n = len(df[df["site"] == site])
            print(f"    {site}: {n} tiles")
    else:
        print(f"  ⚠ Tile index not found: {tile_index}")

    # ══════════════════════════════════════════════════════════
    # 4. MODEL / TRAINING
    # ══════════════════════════════════════════════════════════
    section("MODEL & TRAINING")

    ckpt_dir = Path(cfg.get("checkpoint_dir", "checkpoints"))

    # Loss history
    loss_csv = ckpt_dir / "loss_history.csv"
    if not loss_csv.exists():
        loss_csv = ckpt_dir / "loss_history_v5.csv"

    if loss_csv.exists():
        loss_df = pd.read_csv(loss_csv)
        folds = sorted(loss_df["fold"].unique())
        fold_bests = []
        for fold in folds:
            fd = loss_df[loss_df["fold"] == fold]
            val_col = "val_mae" if "val_mae" in fd.columns else "val_loss"
            best = fd[val_col].min()
            epochs_run = len(fd)
            fold_bests.append({
                "fold": int(fold),
                "best_val_mae": round(best, 4),
                "epochs_run": epochs_run,
            })

        vals = [f["best_val_mae"] for f in fold_bests]
        training_stats = {
            "k_folds": len(folds),
            "fold_results": fold_bests,
            "mean_val_mae": round(np.mean(vals), 4),
            "std_val_mae": round(np.std(vals), 4),
            "total_epochs_run": sum(f["epochs_run"] for f in fold_bests),
        }
        stats["training"] = training_stats

        print(f"  Folds: {len(folds)}")
        for fb in fold_bests:
            print(f"    Fold {fb['fold']}: best val MAE = {fb['best_val_mae']:.4f} "
                  f"({fb['epochs_run']} epochs)")
        print(f"  Mean: {np.mean(vals):.4f} ± {np.std(vals):.4f}")
    else:
        print(f"  ⚠ Loss history not found")

    # Checkpoint file sizes
    for ckpt_name in ["ddt_best_global.pt", "ddt_best_v5_global.pt", "ddt_best.pt"]:
        ckpt_path = ckpt_dir / ckpt_name
        if ckpt_path.exists():
            size_mb = ckpt_path.stat().st_size / 1e6
            stats.setdefault("checkpoints", {})[ckpt_name] = {
                "size_mb": round(size_mb, 1)
            }
            print(f"  Checkpoint: {ckpt_name} ({size_mb:.0f} MB)")

            # Try loading metadata without full model
            try:
                import torch
                meta = torch.load(ckpt_path, map_location="cpu",
                                  weights_only=False)
                if "fold" in meta:
                    stats["checkpoints"][ckpt_name]["fold"] = meta["fold"]
                    stats["checkpoints"][ckpt_name]["epoch"] = meta["epoch"] + 1
                    stats["checkpoints"][ckpt_name]["best_val"] = round(meta["best_val"], 4)
                    print(f"    fold={meta['fold']} epoch={meta['epoch']+1} "
                          f"val={meta['best_val']:.4f}")
                # Count parameters
                n_params = sum(v.numel() for v in meta["model_state"].values())
                stats["checkpoints"][ckpt_name]["parameters"] = n_params
                stats["checkpoints"][ckpt_name]["parameters_millions"] = round(n_params / 1e6, 1)
                print(f"    Parameters: {n_params/1e6:.1f}M")
            except Exception as e:
                print(f"    (Could not read metadata: {e})")

    # ══════════════════════════════════════════════════════════
    # 5. EVALUATION RESULTS
    # ══════════════════════════════════════════════════════════
    section("EVALUATION")

    eval_csv = ckpt_dir / "eval_results.csv"
    if not eval_csv.exists():
        eval_csv = ckpt_dir / "eval_results_v5.csv"

    if eval_csv.exists():
        eval_df = pd.read_csv(eval_csv)

        size_class = eval_df[eval_df["grouping"] == "size_class"]
        site_eval = eval_df[eval_df["grouping"] == "site"]

        eval_stats = {
            "by_size_class": {},
            "by_site": {},
        }

        print(f"\n  {'Class':<12} {'Prec':>7} {'Rec':>7} {'F1':>7} "
              f"{'TP':>5} {'FP':>5} {'FN':>5}")
        print(f"  {'-'*12} {'-'*7} {'-'*7} {'-'*7} {'-'*5} {'-'*5} {'-'*5}")

        for _, row in size_class.iterrows():
            lbl = row["label"]
            eval_stats["by_size_class"][lbl] = {
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
            }
            disp = lbl.upper() if lbl == "overall" else lbl.capitalize()
            print(f"  {disp:<12} {row['precision']:>7.3f} {row['recall']:>7.3f} "
                  f"{row['f1']:>7.3f} {int(row['tp']):>5} "
                  f"{int(row['fp']):>5} {int(row['fn']):>5}")

        print()
        for _, row in site_eval.iterrows():
            lbl = row["label"]
            eval_stats["by_site"][lbl] = {
                "precision": row["precision"],
                "recall": row["recall"],
                "f1": row["f1"],
                "tp": int(row["tp"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
            }
            print(f"  {lbl:<20} F1={row['f1']:.3f}")

        stats["evaluation"] = eval_stats
    else:
        print(f"  ⚠ Eval results not found")

    # ══════════════════════════════════════════════════════════
    # 6. INFERENCE OUTPUTS
    # ══════════════════════════════════════════════════════════
    section("INFERENCE OUTPUTS")

    infer_dir = Path(cfg.get("inference_dir", "outputs/inference"))

    # Crown GeoPackage
    for gpkg_name in ["edmonds_crowns_v5.gpkg", "edmonds_crowns.gpkg"]:
        gpkg_path = infer_dir / gpkg_name
        if gpkg_path.exists():
            gpkg_mb = gpkg_path.stat().st_size / 1e6
            print(f"  GeoPackage: {gpkg_name} ({gpkg_mb:.0f} MB)")

            gdf = gpd.read_file(gpkg_path)
            crown_stats = {
                "file": gpkg_name,
                "file_size_mb": round(gpkg_mb, 1),
                "total_crowns": len(gdf),
                "total_area_m2": round(gdf["area_m2"].sum(), 1),
                "total_area_ha": round(gdf["area_m2"].sum() / 10_000, 1),
                "median_area_m2": round(gdf["area_m2"].median(), 2),
                "mean_area_m2": round(gdf["area_m2"].mean(), 2),
                "median_diameter_m": round(gdf["diameter_m"].median(), 2),
                "mean_diameter_m": round(gdf["diameter_m"].mean(), 2),
            }

            if "size_class" in gdf.columns:
                for sc in ["small", "medium", "large"]:
                    sub = gdf[gdf["size_class"] == sc]
                    crown_stats[f"{sc}_count"] = len(sub)
                    crown_stats[f"{sc}_pct"] = round(100 * len(sub) / len(gdf), 1)

            stats["inference_crowns"] = crown_stats

            print(f"  Total crowns:      {len(gdf):,}")
            print(f"  Total area:        {crown_stats['total_area_ha']:.1f} ha")
            print(f"  Median area:       {crown_stats['median_area_m2']:.1f} m²")
            print(f"  Median diameter:   {crown_stats['median_diameter_m']:.1f} m")
            if "small_count" in crown_stats:
                print(f"  Small:  {crown_stats['small_count']:>8,} ({crown_stats['small_pct']:.1f}%)")
                print(f"  Medium: {crown_stats['medium_count']:>8,} ({crown_stats['medium_pct']:.1f}%)")
                print(f"  Large:  {crown_stats['large_count']:>8,} ({crown_stats['large_pct']:.1f}%)")
            break

    # DTM raster
    for dtm_name in ["edmonds_dtm_v5.tif", "edmonds_dtm.tif"]:
        dtm_path = infer_dir / dtm_name
        if dtm_path.exists():
            dtm_mb = dtm_path.stat().st_size / 1e6
            with rasterio.open(dtm_path) as src:
                stats["inference_dtm"] = {
                    "file": dtm_name,
                    "file_size_mb": round(dtm_mb, 1),
                    "width_px": src.width,
                    "height_px": src.height,
                    "dtype": src.dtypes[0],
                }
            print(f"\n  DTM: {dtm_name} ({dtm_mb:.0f} MB)")
            break

    # Stats CSV
    for stats_name in ["edmonds_crown_stats_v5.csv", "edmonds_crown_stats.csv"]:
        stats_path = infer_dir / stats_name
        if stats_path.exists():
            sdf = pd.read_csv(stats_path)
            city_stats = dict(zip(sdf["metric"], sdf["value"]))
            stats["city_stats"] = city_stats
            print(f"\n  City-level stats from {stats_name}:")
            for k, v in city_stats.items():
                print(f"    {k}: {v}")
            break

    # ══════════════════════════════════════════════════════════
    # 7. BOUNDARY
    # ══════════════════════════════════════════════════════════
    section("CITY BOUNDARY")

    boundary_path = cfg.get("boundary_path")
    if boundary_path and Path(boundary_path).exists():
        bnd = gpd.read_file(boundary_path)
        bnd_utm = bnd.to_crs("EPSG:32610")  # UTM 10N for accurate area
        area_m2 = bnd_utm.union_all().area
        area_ha = area_m2 / 10_000
        area_sqmi = area_m2 / 2.59e6

        bounds = bnd.to_crs("EPSG:3857").total_bounds
        extent_x_m = bounds[2] - bounds[0]
        extent_y_m = bounds[3] - bounds[1]

        boundary_stats = {
            "area_m2": round(area_m2, 0),
            "area_ha": round(area_ha, 1),
            "area_sq_miles": round(area_sqmi, 2),
            "crs": str(bnd.crs),
            "bbox_extent_m": f"{extent_x_m:.0f} × {extent_y_m:.0f}",
        }

        # Canopy cover (if we have crown data)
        if "inference_crowns" in stats:
            cover_pct = 100 * stats["inference_crowns"]["total_area_m2"] / area_m2
            boundary_stats["canopy_cover_pct"] = round(cover_pct, 1)
            print(f"  Canopy cover:    {cover_pct:.1f}%")

        stats["boundary"] = boundary_stats

        print(f"  Area:            {area_ha:.1f} ha ({area_sqmi:.2f} sq mi)")
        print(f"  Bbox extent:     {extent_x_m:.0f} × {extent_y_m:.0f} m")
    else:
        print(f"  ⚠ Boundary not found: {boundary_path}")

    # ══════════════════════════════════════════════════════════
    # 8. REPOSITORY SIZE
    # ══════════════════════════════════════════════════════════
    section("REPOSITORY FILES")

    script_path = Path("scripts/pipeline.py")
    if script_path.exists():
        with open(script_path) as f:
            lines = f.readlines()
        stats["pipeline_script"] = {
            "lines_of_code": len(lines),
            "file_size_kb": round(script_path.stat().st_size / 1024, 1),
        }
        print(f"  pipeline.py: {len(lines)} lines ({script_path.stat().st_size/1024:.1f} KB)")

    # ══════════════════════════════════════════════════════════
    # SUMMARY FOR README
    # ══════════════════════════════════════════════════════════
    section("README-READY SUMMARY")

    print("\nCopy these values into your README:\n")

    if "full_image" in stats and stats["full_image"]:
        fi = stats["full_image"]
        print(f"  Image dimensions:    {fi['width_px']:,} × {fi['height_px']:,} px")
        print(f"  Image resolution:    {fi['pixel_size_cm']} cm/pixel")
        print(f"  Image extent:        {fi['extent_km_str']}")
        print(f"  File size (disk):    {fi['file_size_gb']} GB")
        print(f"  Decompressed size:   {fi['uncompressed_gb']} GB")
        print(f"  Float32 in memory:   {fi['float32_memory_gb']} GB")

    if "training_sites_summary" in stats:
        ts = stats["training_sites_summary"]
        print(f"  Training sites:      {ts['total_sites']} ({ts['positive_sites']} positive, {ts['negative_sites']} negative)")
        print(f"  Total annotations:   {ts['total_crowns']} crowns")

    if "tiles" in stats:
        t = stats["tiles"]
        print(f"  Training tiles:      {t['train_tiles']}")
        print(f"  Test tiles:          {t['test_tiles']}")
        print(f"  Total tiles:         {t['total_tiles']}")

    if "training" in stats:
        tr = stats["training"]
        print(f"  CV folds:            {tr['k_folds']}")
        print(f"  Val MAE:             {tr['mean_val_mae']} ± {tr['std_val_mae']}")

    if "evaluation" in stats and "by_size_class" in stats["evaluation"]:
        ev = stats["evaluation"]["by_size_class"]
        if "overall" in ev:
            ov = ev["overall"]
            print(f"  Overall F1:          {ov['f1']}")
            print(f"  Overall precision:   {ov['precision']}")
            print(f"  Overall recall:      {ov['recall']}")
        for sc in ["small", "medium", "large"]:
            if sc in ev:
                print(f"  {sc.capitalize()} F1:          {ev[sc]['f1']}")

    if "inference_crowns" in stats:
        ic = stats["inference_crowns"]
        print(f"  Crowns detected:     {ic['total_crowns']:,}")
        print(f"  Total crown area:    {ic['total_area_ha']} ha")
        print(f"  Median crown area:   {ic['median_area_m2']} m²")
        print(f"  Median diameter:     {ic['median_diameter_m']} m")

    if "boundary" in stats:
        b = stats["boundary"]
        print(f"  City area:           {b['area_ha']} ha ({b['area_sq_miles']} sq mi)")
        if "canopy_cover_pct" in b:
            print(f"  Canopy cover:        {b['canopy_cover_pct']}%")

    if "checkpoints" in stats:
        for name, ck in stats["checkpoints"].items():
            if "parameters_millions" in ck:
                print(f"  Model parameters:    {ck['parameters_millions']}M")
                break

    # ══════════════════════════════════════════════════════════
    # SAVE
    # ══════════════════════════════════════════════════════════
    out_dir = Path(cfg.get("inference_dir", "outputs"))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "repo_stats.yaml"

    # Clean stats for YAML serialization
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(out_path, "w") as f:
        yaml.dump(clean(stats), f, default_flow_style=False, sort_keys=False)

    print(f"\n✓ Full stats saved to: {out_path}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Gather repository stats for README accuracy")
    parser.add_argument("--config", type=str, default="configs/colab.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    gather_stats(cfg)


if __name__ == "__main__":
    main()
