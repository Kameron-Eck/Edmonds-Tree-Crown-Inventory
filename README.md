# 🌲 Edmonds Urban Tree Crown Detection

**Individual tree crown detection from 7.5 cm aerial imagery using Deep Distance Transforms and U-Net semantic segmentation.**

Built for the [City of Edmonds Climate Advisory Board](https://www.edmondswa.gov/) to create the city's first automated urban tree inventory.

![Pipeline Overview](docs/pipeline_diagram.png)

---

## Highlights

| Metric | Value |
|--------|-------|
| **Crowns detected** | 180,000+ across Edmonds city limits |
| **Model F1 score** | 0.799 (large crowns), 0.74 overall |
| **Imagery resolution** | 7.5 cm/pixel RGB |
| **Training data** | 530 hand-annotated 512×512 tiles across 5 sites |
| **Architecture** | U-Net + ResNet-101 encoder (44M parameters) |
| **Validation** | 5-fold cross-validation with early stopping |
| **Inference speed** | ~2 hours for full city (26,000×35,000 px) on A100 |

## Approach

This project implements the **Deep Distance Transform (DDT)** method for individual tree crown detection:

1. **Distance Transform Labels** — For each annotated crown polygon, compute a normalized distance field where pixel values range from 100 (at centroid) to 1 (at boundary) to 0 (background). This continuous regression target preserves crown shape and size information that binary masks lose.

2. **U-Net Regression** — Train a U-Net with pretrained ResNet-101 encoder to predict the distance transform from RGB input. The model learns both "where are trees" and "where are tree centers" simultaneously.

3. **Watershed Segmentation** — At inference time, local maxima in the predicted distance field serve as crown seeds. Watershed flooding from these seeds delineates individual crown polygons, even for touching/overlapping canopies.

This approach was originally developed by [Hickman et al. (2021)](https://doi.org/10.1016/j.rse.2021.112641) for tropical forests and has been adapted here for Pacific Northwest urban forest conditions.

```
RGB Aerial Image → U-Net → Predicted Distance Transform → Watershed → Crown Polygons (GeoPackage)
```

## Repository Structure

```
edmonds-tree-crown-detection/
├── scripts/
│   └── pipeline.py          # Complete end-to-end pipeline (single file)
├── configs/
│   ├── default.yaml          # Default hyperparameters
│   └── colab.yaml            # Google Colab overrides
├── docs/
│   ├── pipeline_diagram.png  # Architecture diagram
│   ├── methodology.md        # Detailed methodology
│   └── results.md            # Full evaluation results
├── data/                     # Not tracked — see Data Setup below
│   ├── photos/               # *_rgb.tif training site imagery
│   ├── polygons/             # Matching .shp crown annotations
│   ├── boundary/             # City boundary shapefile
│   └── ...
├── checkpoints/              # Not tracked — model weights
├── outputs/                  # Not tracked — inference results
├── requirements.txt
├── LICENSE
└── README.md
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Data

Place training data following this naming convention:

```
data/photos/Forest_1_rgb.tif       ← RGB imagery (EPSG:3857, 7.5cm)
data/polygons/Forest_1.shp         ← Crown polygon annotations
data/photos/Negative_Parking_rgb.tif  ← True negative (no .shp needed)
```

The pipeline auto-discovers all `*_rgb.tif` files and pairs them with matching shapefiles. Sites without a `.shp` are treated as true negatives.

### 3. Run Full Pipeline

```bash
# All stages: discover → inspect → preprocess → dtm → tile → train → evaluate
python scripts/pipeline.py --config configs/default.yaml

# Specific stages only
python scripts/pipeline.py --config configs/default.yaml --stages discover preprocess dtm tile

# Train and evaluate
python scripts/pipeline.py --config configs/default.yaml --stages train evaluate

# Full-city inference (requires full_image_path in config)
python scripts/pipeline.py --config configs/default.yaml --stages infer
```

### 4. Google Colab

```python
# Mount Drive and install deps
from google.colab import drive
drive.mount('/content/drive')
!pip install segmentation-models-pytorch albumentations timm==0.9.7 -q

# Run with Colab config
!python scripts/pipeline.py --config configs/colab.yaml --stages train evaluate
```

## Pipeline Stages

| Stage | Description | Outputs |
|-------|-------------|---------|
| `discover` | Auto-find training sites from file naming convention | Site inventory |
| `inspect` | QA check on raster/shapefile properties, CRS, band stats | Console report |
| `preprocess` | CRS reprojection, geometry repair, sliver removal | Clean GeoDataFrames |
| `dtm` | Centroid-based normalized distance transform generation | `*_dtm.tif` per site |
| `tile` | 512×512 paired RGB/DTM tiles with stratified train/test split | `tile_index.csv` |
| `train` | 5-fold CV with weighted sampling, mixed precision, early stopping | Model checkpoints |
| `evaluate` | Watershed → IoU matching → F1 by size class and site | `eval_results.csv` |
| `infer` | Streaming inference over full city image → crown GeoPackage | `.gpkg` + stats |

## Key Design Decisions

**Why distance transforms instead of instance segmentation (Mask R-CNN)?**
Distance transforms encode crown shape as a continuous field. This means the U-Net only needs to solve a regression problem, not the harder instance segmentation problem. Watershed then handles instance separation cleanly at post-processing time, without anchor boxes or NMS.

**Why 5-fold cross-validation?**
With only ~530 annotated tiles across 5 sites, a single train/test split risks overfitting to the specific test partition. 5-fold CV gives a more robust estimate of generalization performance (MAE 6.91 ± 0.53 across folds).

**Why weighted sampling?**
Training sites vary dramatically in size (Forest_1: 276 tiles, Forest_2: 39 tiles). Without balancing, the model would overfit to the largest site. `WeightedRandomSampler` ensures each site contributes equally per epoch.

**Why streaming inference?**
The full Edmonds image is ~26,000 × 35,000 pixels (~3.5 GB). Loading it entirely into memory is impractical. The pipeline reads tiles on-the-fly, predicts in batches, and writes center crops directly to a BigTIFF with LZW compression.

## Model Architecture

```
Input: RGB tile (3 × 512 × 512)
    │
    ▼
┌─────────────────────────────┐
│  ResNet-101 Encoder          │  ← ImageNet pretrained
│  (stages 1-5, 44M params)   │
└──────────┬──────────────────┘
           │ skip connections
           ▼
┌─────────────────────────────┐
│  U-Net Decoder               │
│  channels: 1024→512→256→128→64
│  + Dropout2d(0.3) per block │
└──────────┬──────────────────┘
           │
           ▼
Output: Distance Transform (1 × 512 × 512)
    values: 0 (background) to 100 (crown center)
```

## Training Details

- **Loss**: L1 (MAE) + L1 weight regularization (λ=1e-6)
- **Optimizer**: AdamW (lr=1e-4, weight decay=1e-4)
- **Scheduler**: Linear warmup (5 epochs) → ReduceLROnPlateau (patience=8)
- **Augmentation**: Flips, rotations, elastic distortion, brightness/contrast, synthetic shadow, fog simulation, coarse dropout
- **Mixed precision**: FP16 forward/backward with GradScaler
- **Gradient clipping**: max_norm=1.0

## Evaluation Results

Detection evaluated with IoU ≥ 0.5 matching after watershed segmentation:

| Crown Size | Precision | Recall | F1 | Notes |
|------------|-----------|--------|-----|-------|
| Small (≤4.9 m²) | — | — | 0.000 | Limited small-crown training data |
| Medium (4.9–15.9 m²) | 0.308 | 0.235 | 0.267 | Suburban ornamentals |
| Large (>15.9 m²) | 0.826 | 0.773 | 0.799 | Mature conifers/broadleaf |
| **Overall** | **0.779** | **0.702** | **0.738** | **Exceeds DDT paper (0.531)** |

> The model significantly outperforms the original DDT paper baseline (F1=0.531). Small crown detection is the primary area for improvement — future annotation campaigns targeting residential ornamental trees and street trees will address this gap.

## Data Requirements

| Input | Format | CRS | Resolution |
|-------|--------|-----|------------|
| Training imagery | GeoTIFF (RGB, uint8) | EPSG:3857 | 7.5 cm |
| Crown annotations | Shapefile (Polygon) | Any (auto-reprojected) | — |
| City boundary | Shapefile (Polygon) | Any (auto-reprojected) | — |
| Full city image | GeoTIFF (RGB, uint8) | EPSG:3857 | 7.5 cm |

Training imagery is 2020 aerial photography provided by the City of Edmonds. Crown annotations were hand-digitized in QGIS from the 7.5 cm imagery.

## Output Formats

| Output | Format | Description |
|--------|--------|-------------|
| Crown polygons | GeoPackage (`.gpkg`) | Individual crown polygons with area, diameter, size class |
| Distance transform | GeoTIFF (`.tif`) | Full-city predicted DTM raster |
| Evaluation metrics | CSV | F1/precision/recall by size class and training site |
| Loss history | CSV | Per-epoch training and validation MAE for all folds |

The GeoPackage is compatible with QGIS, ArcGIS, and any OGC-compliant GIS platform.

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings to tune:

```yaml
# For different GPU hardware
batch_size: 30        # A100: 30, V100: 16, T4: 8
num_workers: 16       # Linux: 16, Colab: 2

# For different imagery
expected_res_cm: 7.5  # Adjust if using different resolution
tile_size: 512        # Match to crown sizes in your area

# For watershed tuning
min_distance: 30      # ↑ merges more, ↓ splits more
dtm_threshold: 10.0   # ↑ stricter crown boundary, ↓ captures more edge pixels
```

## Future Work

- **Expand training data**: Add 200–300 tiles in residential areas and street tree corridors to improve small/medium crown detection
- **Post-processing masks**: Building footprint, water body, and road ROW masks to reduce false positives
- **Temporal analysis**: Re-run on future imagery (2024+) for crown loss/gain tracking
- **Canopy equity analysis**: Intersect with census tract demographics for environmental justice mapping
- **Species-level classification**: Explore adding spectral indices or LiDAR fusion for conifer/deciduous separation

## References

- Hickman, S., et al. (2021). "Individual tree crown delineation from airborne laser scanning for diseased larch tree detection." *Remote Sensing of Environment*, 263, 112641.
- Ball, J.G.C., et al. (2023). "Accurate delineation of individual tree crowns in tropical forests from aerial RGB imagery using Mask R-CNN." *Remote Sensing in Ecology and Conservation*.
- detectree2: [GitHub](https://github.com/PatBall1/detectree2) — Deep learning for tree crown delineation

## Author

**Kam** — Environmental Specialist & GIS Analyst
Edmonds Climate Advisory Board

M.S. Culture & Environmental Resource Management (GIS Certification), Central Washington University
B.S. Environmental Science & Public Policy

## License

MIT License — see [LICENSE](LICENSE) for details.

Training data and model weights are not included in this repository due to file size and data sharing agreements with the City of Edmonds.
