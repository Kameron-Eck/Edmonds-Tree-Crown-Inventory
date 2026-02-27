# Methodology

## Deep Distance Transform (DDT) for Urban Tree Crown Detection

### Problem Statement

Individual tree crown detection in urban environments presents unique challenges compared to forestry applications:

- **Mixed species composition**: Pacific Northwest cities contain both native conifers (Douglas fir, Western red cedar) with conical crowns and deciduous species (bigleaf maple, red alder) with rounded crowns
- **Urban context**: Trees are interspersed with buildings, roads, vehicles, and other impervious surfaces that can confuse detection models
- **Scale variation**: Crown areas range from <5 m² (young ornamentals) to >100 m² (mature conifers), requiring multi-scale feature extraction
- **Touching canopies**: Adjacent crowns frequently overlap in aerial nadir imagery, making instance separation critical

### Why Distance Transforms?

Traditional approaches to individual tree detection include:

1. **Object detection** (Mask R-CNN, YOLO): Predict bounding boxes + masks. Struggles with irregular crown shapes and requires anchor tuning per dataset.
2. **Semantic segmentation**: Binary crown/background classification. Cannot separate touching crowns.
3. **Instance segmentation**: Combines detection + segmentation. Heavy architectures, complex loss functions.

The **Deep Distance Transform** approach offers an elegant middle ground:

- The model solves a **regression** problem (predict continuous distance values), which is simpler than instance segmentation
- The distance field naturally encodes **crown shape, size, and center location** in a single output channel
- **Watershed segmentation** at post-processing time handles instance separation without learned components
- The approach is **architecture-agnostic** — any encoder-decoder model works (we use U-Net)

### Distance Transform Construction

For each annotated crown polygon:

```
1. Rasterize the polygon to a binary pixel mask
2. Compute the centroid of the polygon geometry
3. For each pixel inside the crown:
   - Calculate Euclidean distance to the centroid (in pixels)
   - Normalize: value = (1 - dist/max_dist) × 99 + 1
4. Background pixels receive value 0
```

This produces a continuous field where:
- **100** = crown centroid (peak)
- **1** = crown boundary (edge)
- **0** = background (no tree)

The normalization to [1, 100] ensures all crowns have the same value range regardless of size, allowing the model to learn a size-invariant representation.

### Data Augmentation Strategy

Augmentations were critical given the limited training data (530 tiles). The pipeline applies:

**Spatial transforms** (applied to RGB and label jointly):
- Horizontal/vertical flip, transpose, 90° rotation
- Random rotation (±45°), shift-scale-rotate
- Grid distortion, elastic transform

**Pixel transforms** (applied to RGB only):
- Gaussian/median blur
- Brightness, contrast, hue, saturation, gamma variation
- Synthetic shadows and fog (simulates time-of-day and atmospheric effects)
- Coarse dropout (random rectangular occlusions)
- Downscale-upscale (simulates lower-resolution imagery)

### Training Strategy

**5-Fold Cross-Validation**: With only 5 training sites, a single split risks biasing toward one site's characteristics. K-fold CV provides robust generalization estimates.

**Weighted Sampling**: Sites vary in tile count (39–276 tiles). Without balancing, the model overfits to the largest site. WeightedRandomSampler ensures equal site contribution per epoch.

**Mixed Precision Training**: FP16 forward/backward passes with GradScaler reduce memory usage and increase throughput by ~2× on modern GPUs.

**Regularization Stack**:
- Decoder dropout (p=0.3): Prevents co-adaptation in decoder features
- L1 weight penalty (λ=1e-6): Encourages sparsity in learned weights
- Gradient clipping (max_norm=1.0): Stabilizes training on noisy gradients
- Early stopping (patience=25): Prevents overfitting past optimal convergence

### Inference Architecture

Full-city inference over a 26,000 × 35,000 pixel image requires careful memory management:

1. **Sliding window**: 512×512 tiles with 256px stride (50% overlap)
2. **Center cropping**: Only the center 256×256 of each prediction is written, avoiding edge artifacts
3. **Reflect padding**: Image borders are padded with reflected pixels
4. **Streaming writes**: Predictions are written directly to a BigTIFF — no full-image array in memory
5. **Batched GPU inference**: 160 tiles per forward pass on A100

### Watershed Post-Processing

The predicted distance transform is converted to individual crown polygons:

1. **Threshold**: Pixels below DTM_THRESHOLD (10.0) are masked as background
2. **Peak detection**: Local maxima with MIN_DISTANCE (30px ≈ 2.2m) separation become crown seeds
3. **Watershed flooding**: Seeds flood downhill (inverted DTM) within the crown mask
4. **Vectorization**: Each watershed basin is converted to a polygon geometry
5. **Size filter**: Crowns below MIN_CROWN_AREA (2.0 m²) are dropped

Chunked processing with 300px border overlap and centroid-based ownership prevents artifacts at chunk boundaries.

### Evaluation Protocol

Following the DDT paper, evaluation uses:

- **IoU ≥ 0.5** matching between predicted and ground truth crown masks
- **Greedy matching**: Each GT crown is matched to the highest-IoU prediction (used once)
- **Size class stratification**: Small (≤4.9 m²), Medium (4.9–15.9 m²), Large (>15.9 m²)
- **Per-site reporting**: Ensures model generalizes across different forest contexts

### Limitations and Known Issues

1. **Small crown detection (F1=0.00)**: Training data is dominated by large conifers. Small ornamental trees in residential areas are underrepresented.
2. **Conifer/deciduous bias**: Three forest training sites are conifer-heavy. Deciduous tree performance may degrade in heavily broadleaf neighborhoods.
3. **Seasonal dependency**: Training data is from leaf-on season. Leaf-off imagery would require retraining.
4. **Building shadows**: Dark shadows from tall buildings can be confused with conifer canopy.
5. **No height information**: Without LiDAR, the model cannot distinguish trees from tall shrubs or green roofs.
