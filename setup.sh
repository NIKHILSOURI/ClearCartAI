#!/bin/bash
# ─── Product Segmenter Setup Script ──────────────────────────
# Run once: bash setup.sh
# Requires: CUDA-capable GPU, conda or Python 3.10+

set -e

echo "============================================"
echo " Product Segmentation Pipeline Setup"
echo "============================================"

# ─── 1. Create conda environment (optional) ──────────────────
if command -v conda &> /dev/null; then
    echo "[1/5] Creating conda environment..."
    conda create -n product-seg python=3.11 -y
    eval "$(conda shell.bash hook)"
    conda activate product-seg
else
    echo "[1/5] Conda not found, using system Python"
fi

# ─── 2. Install PyTorch with CUDA ────────────────────────────
echo "[2/5] Installing PyTorch..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# ─── 3. Install SAM2 ─────────────────────────────────────────
echo "[3/5] Installing SAM2 from source..."
pip install git+https://github.com/facebookresearch/sam2.git

# ─── 4. Install remaining dependencies ───────────────────────
echo "[4/5] Installing dependencies..."
pip install transformers>=4.35.0 \
            Pillow>=10.0.0 \
            opencv-python>=4.8.0 \
            matplotlib>=3.7.0 \
            scipy>=1.11.0 \
            tqdm>=4.65.0 \
            numpy>=1.24.0

# ─── 5. Download model checkpoints ───────────────────────────
echo "[5/5] Pre-downloading model weights (this may take a few minutes)..."

python -c "
from transformers import AutoImageProcessor, AutoModel
print('Downloading DINOv2 ViT-L/14 with registers...')
AutoImageProcessor.from_pretrained('facebook/dinov2-vitl14-reg')
AutoModel.from_pretrained('facebook/dinov2-vitl14-reg')
print('DINOv2 downloaded successfully!')

print('Downloading SAM2.1 Hiera Large...')
from sam2.sam2_image_predictor import SAM2ImagePredictor
SAM2ImagePredictor.from_pretrained('facebook/sam2.1-hiera-large')
print('SAM2 downloaded successfully!')
"

echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo ""
echo "Usage (run from project root):"
echo "  # Interactive mode:"
echo "  python scripts/run_interactive.py --image-dir ./my_360_captures/ --output ./results/"
echo ""
echo "  # Batch mode:"
echo "  python scripts/run_batch.py --image-dir ./my_360_captures/ --reference img_001.jpg --point 250 300"
echo ""
