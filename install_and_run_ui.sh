#!/usr/bin/env bash
# ============================================================
# EAN Product Segmentation - Install and Run Labeling UI
# Linux / macOS. Run from project root: bash install_and_run_ui.sh
# ============================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================"
echo " EAN Product Segmentation - Install & Run UI"
echo "============================================"
echo ""

# --- 1. Virtual environment ---
if [ ! -d "venv" ]; then
    echo "[1/5] Creating virtual environment..."
    python3 -m venv venv 2>/dev/null || python -m venv venv
else
    echo "[1/5] Using existing venv"
fi

PIP="$PROJECT_ROOT/venv/bin/pip"
PYTHON="$PROJECT_ROOT/venv/bin/python"

if [ ! -x "$PIP" ]; then
    echo "ERROR: venv not found or broken. Remove 'venv' and run again."
    exit 1
fi

# --- 2. Dependencies ---
echo "[2/5] Installing dependencies from requirements.txt..."
"$PIP" install -q -r requirements.txt

# --- 3. PyTorch with CUDA ---
echo "[3/5] Installing PyTorch with CUDA 12.1..."
"$PIP" uninstall -y torch torchvision torchaudio 2>/dev/null || true
"$PIP" install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121

# --- 4. SAM2 ---
echo "[4/5] Installing SAM2 from source..."
"$PIP" install -q "git+https://github.com/facebookresearch/sam2.git"

# --- 5. Verify CUDA ---
echo "[5/5] Verifying CUDA..."
"$PYTHON" -c "import torch; print('CUDA available:', torch.cuda.is_available())"

echo ""
echo "============================================"
echo " Install complete!"
echo "============================================"
echo ""

# --- Run UI ---
RUN_UI=true
for arg in "$@"; do
    if [ "$arg" = "--install-only" ]; then
        RUN_UI=false
        break
    fi
done

if [ "$RUN_UI" = false ]; then
    echo "Install only (--install-only). To run the UI later:"
    echo "  source venv/bin/activate"
    echo "  export DATABASE_URL='postgresql://user:pass@localhost:5432/dbname'"
    echo "  python tools/label_ui_gradio.py"
    echo ""
    exit 0
fi

if [ -z "${DATABASE_URL:-}" ]; then
    echo "WARNING: DATABASE_URL is not set. Set it for the labeling UI, e.g.:"
    echo "  export DATABASE_URL='postgresql://user:password@localhost:5432/ean_labels'"
    echo ""
    read -r -p "Continue anyway? (y/N) " continue
    if [ "$continue" != "y" ] && [ "$continue" != "Y" ]; then
        echo "Set DATABASE_URL and run: $PYTHON tools/label_ui_gradio.py"
        exit 0
    fi
fi

echo "Starting Gradio labeling UI on http://localhost:7860 ..."
echo "Press Ctrl+C to stop."
echo ""
exec "$PYTHON" tools/label_ui_gradio.py
