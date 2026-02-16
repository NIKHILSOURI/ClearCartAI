# EAN Vision System (SAM 3 Click Segmentation -> Packaging Classifier -> Product Classifier -> EAN Lookup)

## What you get
This repo implements the workflow you requested:

1) Labeling UI: Click on the product in a photo, SAM 3 segments it, the tool saves:
   - mask PNG
   - cropped product image (white background JPG)
   - a JSONL record (packaging + product_name + optional session_id)

2) Dataset builder: Converts records.jsonl into Ultralytics classification datasets:
   - data/datasets/packaging_cls (classes = packaging)
   - data/datasets/products_cls (classes = product_name)
   - data/datasets/products_cls_by_pack/<packaging> (recommended at 10k products)

3) Training scripts:
   - packaging router (YOLO26 classification)
   - product model(s) (global or per packaging)

4) Inference:
   - image -> auto-crop with SAM 3 (center click) -> packaging -> product_name -> EAN lookup (CSV)
   - FastAPI endpoint: POST /predict-ean

## Quick Start

### Setup (First Time Only)
```powershell
# Windows
.\setup.ps1

# Linux/Mac
chmod +x setup.sh
./setup.sh
```

### Start Labeling Tool
```powershell
# Quick start (recommended)
.\quick_start.ps1

# Or manual start
$env:PYTHONPATH = "path\to\ean_system"
.\.venv\Scripts\python.exe tools\label_ui_gradio.py
```

**Important**: 
- Disable VPN before starting
- Add images to `raw_pictures/` folder
- Open browser to `http://127.0.0.1:7860`

### Usage
1. Click "Load Next" to browse images
2. Click on product center
3. Click "Run SAM 3" (wait 8-10 seconds)
4. Select packaging type and enter product name
5. Click "Save Record" to save and load next

See [CHANGELOG.md](CHANGELOG.md) for recent improvements and [DEPLOYMENT.md](DEPLOYMENT.md) for delivery instructions.

### 3) Build datasets from labels
```bash
python scripts/01_build_datasets.py
```

### 4) Train packaging router
```bash
python scripts/02_train_packaging.py
```

### 5) Train product models
Per packaging (recommended):
```bash
python scripts/03_train_products.py --mode by_pack
```

Or one global model:
```bash
python scripts/03_train_products.py --mode global
```

### 6) EAN lookup mapping (external)
Generate a template from your product_name labels:
```bash
python scripts/00_make_product_to_ean_template.py
```

Fill EANs in:
```
data/mappings/product_to_ean.csv
```

### 7) Run API
```bash
uvicorn src.ean.api_fastapi:app --host 0.0.0.0 --port 8000
```

Test:
```bash
curl -X POST "http://localhost:8000/predict-ean" -F "file=@data/raw/example.jpg"
```

## Notes
- Training/inference quality is best if you classify on the segmented crop, not the raw image.
- Use confidence thresholds to return "unknown" rather than wrong EAN.
