# 360° Product Segmentation Pipeline

## SAM2 + DINOv2 Cross-Image Instance Segmentation System

**Mission**: Capture a supermarket product (milk carton, etc.) from a 360° camera across multiple images, segment it interactively on one image using SAM2, then automatically find and segment the **same product instance** on all other images.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    360° PRODUCT SEGMENTER                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: Interactive Segmentation (Reference Image)        │
│  ┌──────────┐    ┌─────────┐    ┌──────────────────┐       │
│  │ 360° Img │───>│  SAM2   │───>│ Reference Mask    │       │
│  │ (select) │    │ (click) │    │ + Bounding Box     │       │
│  └──────────┘    └─────────┘    └────────┬─────────┘       │
│                                          │                  │
│  Phase 2: Feature Extraction             │                  │
│  ┌──────────────────────────────────────┐│                  │
│  │ DINOv2 ViT-L/14                      ││                  │
│  │ ┌────────────┐  ┌─────────────────┐  ││                  │
│  │ │ Patch      │  │ Foreground      │  ││                  │
│  │ │ Embeddings │─>│ Feature Average │──┘│                  │
│  │ │ (frozen)   │  │ (FFA)           │   │                  │
│  │ └────────────┘  └────────┬────────┘   │                  │
│  └───────────────────────────┼───────────┘                  │
│                              │                              │
│  Phase 3: Cross-Image Matching                              │
│  ┌───────────────────────────┼───────────────────────┐      │
│  │ For each target image:    │                       │      │
│  │ ┌──────────┐  ┌──────────▼───────┐  ┌──────────┐ │      │
│  │ │ SAM2     │  │ DINOv2 FFA       │  │ Cosine   │ │      │
│  │ │ Auto-    │─>│ per proposal     │─>│ Matching │ │      │
│  │ │ Segment  │  │                  │  │ + NMS    │ │      │
│  │ └──────────┘  └──────────────────┘  └────┬─────┘ │      │
│  └──────────────────────────────────────────┼───────┘      │
│                                             │               │
│  Phase 4: Output                            │               │
│  ┌──────────────────────────────────────────▼──────┐        │
│  │ • Per-image segmentation masks (PNG alpha)      │        │
│  │ • Bounding boxes + confidence scores            │        │
│  │ • Cropped product cutouts (transparent BG)      │        │
│  │ • Combined visualization gallery                │        │
│  └─────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

---

## Pipeline Design Rationale

### Why SAM2 + DINOv2 (not SAM2 alone)?

SAM2 is exceptional at **segmentation** — given a prompt (click, box), it produces
pixel-perfect masks. But SAM2's internal features are **not semantically rich enough
to distinguish instances across images**. Research (NIDS-Net, CNOS) confirms that
SAM's encoder lacks discriminative power for cross-image matching.

**DINOv2** fills this gap. Its self-supervised patch embeddings produce features where:
- The same product from different angles clusters tightly
- Different products (even visually similar ones) separate cleanly
- Works zero-shot with no training required

### The FFA (Foreground Feature Averaging) Trick

Instead of using DINOv2's global CLS token (which mixes foreground + background),
we use SAM2's mask to **select only foreground patch tokens** from DINOv2, then
average them. This produces a clean embedding of just the product, invariant to
background clutter.

---

## System Requirements

| Component        | Minimum              | Recommended          |
|-----------------|---------------------|---------------------|
| GPU             | NVIDIA 8GB VRAM     | NVIDIA 16GB+ VRAM   |
| CUDA            | 11.8+               | 12.1+               |
| Python          | 3.10+               | 3.11                |
| RAM             | 16GB                | 32GB                |
| Storage         | 10GB (models)       | 20GB                |

---

## File Structure

```
product-segmenter/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.sh                     # One-shot setup script
├── config.py                    # All configuration in one place
├── models/
│   ├── model_loader.py          # Lazy model loading & caching
│   └── checkpoints/             # Downloaded model weights
├── core/
│   ├── __init__.py
│   ├── sam2_segmenter.py        # SAM2 interactive + auto segmentation
│   ├── dinov2_embedder.py       # DINOv2 feature extraction + FFA
│   ├── matcher.py               # Cross-image instance matching
│   └── pipeline.py              # End-to-end orchestration
├── ui/
│   ├── interactive_selector.py  # Click-to-segment UI (matplotlib)
│   └── gallery_viewer.py        # Results visualization
├── utils/
│   ├── image_utils.py           # Loading, resizing, 360° handling
│   ├── mask_utils.py            # Mask operations, NMS, refinement
│   └── export.py                # Export masks, cutouts, COCO format
├── run_interactive.py           # Main entry: interactive mode
├── run_batch.py                 # Main entry: batch/automated mode
└── outputs/                     # Generated results
```

---

## Detailed Module Specifications

### Module 1: `config.py` — Central Configuration

All tunable parameters in one place. Designed so you can override via CLI or env vars.

### Module 2: `models/model_loader.py` — Model Management

Handles downloading SAM2 and DINOv2 checkpoints, building models with correct configs,
and caching loaded models in memory to avoid redundant GPU loads.

### Module 3: `core/sam2_segmenter.py` — SAM2 Segmentation

Two modes:
1. **Interactive**: User clicks on product → SAM2 produces mask
2. **Auto-generate**: SAM2 generates all possible masks in an image (for matching)

Uses `SAM2ImagePredictor` for single-image tasks. The auto-generator uses
`SAM2AutomaticMaskGenerator` with tuned parameters for retail products.

### Module 4: `core/dinov2_embedder.py` — DINOv2 Feature Extraction

The key innovation module. Given an image + mask:
1. Runs image through DINOv2 ViT-L/14 backbone
2. Extracts all patch embeddings (not just CLS token)
3. Maps the SAM2 mask onto the patch grid (14×14 or 16×16)
4. Averages only the patch embeddings falling inside the mask
5. L2-normalizes the result → **instance embedding vector**

### Module 5: `core/matcher.py` — Cross-Image Instance Matching

For each target image:
1. Generate candidate proposals (SAM2 auto-segment)
2. Compute FFA embedding for each proposal
3. Compare each proposal embedding to reference embedding via cosine similarity
4. Apply threshold + NMS to get final matches

### Module 6: `core/pipeline.py` — End-to-End Orchestration

Ties everything together. Handles the full flow from "here are my 360° images"
to "here are segmented product masks on every image."

---

## Key Parameters to Tune

| Parameter                    | Default  | Description                                           |
|-----------------------------|---------|-------------------------------------------------------|
| `SIMILARITY_THRESHOLD`       | 0.65    | Min cosine similarity to consider a match             |
| `SAM2_POINTS_PER_SIDE`       | 32      | Auto-mask generator density (higher = more proposals) |
| `SAM2_PRED_IOU_THRESH`       | 0.86    | Minimum predicted IoU for auto-masks                  |
| `SAM2_STABILITY_SCORE_THRESH`| 0.92    | Minimum mask stability score                          |
| `MIN_MASK_AREA_RATIO`        | 0.001   | Skip masks smaller than this fraction of image        |
| `MAX_MASK_AREA_RATIO`        | 0.5     | Skip masks larger than this fraction (background)     |
| `NMS_IOU_THRESHOLD`          | 0.5     | NMS overlap threshold for duplicate removal           |
| `DINOV2_MODEL`               | ViT-L/14| DINOv2 backbone size                                  |
| `SAM2_CHECKPOINT`            | hiera_large | SAM2 model size                                   |
| `TOP_K_MATCHES`              | 3       | Max matches per target image                          |

---

## Usage Scenarios

### Scenario A: Interactive Mode (Primary)

```bash
python run_interactive.py \
    --images ./360_captures/*.jpg \
    --output ./results/
```

1. Displays first image → you click on the product
2. SAM2 segments it → you confirm or refine with more clicks
3. Pipeline automatically processes all remaining images
4. Outputs: masks, cutouts, visualization gallery

### Scenario B: Batch Mode (Automated)

```bash
python run_batch.py \
    --images ./360_captures/*.jpg \
    --reference ./360_captures/img_001.jpg \
    --ref-bbox 120,80,340,450 \
    --output ./results/
```

Provide reference image + bounding box coordinates. No UI needed.
Good for automation, CI/CD, large batches.

### Scenario C: API Integration

```python
from core.pipeline import ProductSegmentationPipeline

pipeline = ProductSegmentationPipeline()
results = pipeline.run(
    image_paths=["img1.jpg", "img2.jpg", ...],
    reference_image="img1.jpg",
    reference_point=(250, 300),  # click coordinate
)

for r in results:
    r.save_mask("mask.png")
    r.save_cutout("cutout.png")
```
