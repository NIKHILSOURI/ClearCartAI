#!/usr/bin/env bash
#
# ClearCartAI — One-shot RunPod (or any pod) setup and run
# Run once per pod: clone repo (if needed), install deps, configure DB, start Gradio.
#
# Usage:
#   bash setup_and_run_pod.sh           # full setup + start Gradio UI
#   bash setup_and_run_pod.sh --install-only   # setup only (no UI start)
#
# Uses /workspace for persistence (clone and DB under /workspace/ClearCartAI).
#
set -e

REPO_URL="https://github.com/NIKHILSOURI/ClearCartAI.git"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${WORKSPACE}/ClearCartAI"

echo "=============================================="
echo "  ClearCartAI — Pod setup and run"
echo "=============================================="

# --- 1) Sanity check: persistence and disk
echo ""
echo "[1/7] Checking workspace and persistence..."
pwd
ls -la "$WORKSPACE" 2>/dev/null || true
df -h | head -5
echo "hello" > "${WORKSPACE}/PERSIST_TEST.txt"
ls -la "${WORKSPACE}/PERSIST_TEST.txt"
echo "  -> Persistence check OK (file created on persistent volume)."

# --- 2) Clone repo if not present
echo ""
echo "[2/7] Ensuring repo is cloned..."
if [ ! -d "$REPO_DIR" ]; then
  echo "  Cloning $REPO_URL into $WORKSPACE ..."
  cd "$WORKSPACE"
  git clone "$REPO_URL"
  cd ClearCartAI
else
  echo "  -> Repo already exists at $REPO_DIR (skipping clone)."
  cd "$REPO_DIR"
fi

# --- 3) Virtual environment
echo ""
echo "[3/7] Virtual environment..."
if [ ! -d "venv" ]; then
  echo "  Creating venv..."
  python3 -m venv venv
fi
# Remove broken pip metadata so "Ignoring invalid distribution ~ip" goes away
rm -rf venv/lib/python3.11/site-packages/'~ip'* 2>/dev/null || true
echo "  Activating venv..."
# shellcheck source=/dev/null
source venv/bin/activate
# Use venv's Python and pip explicitly (avoid system python/pip)
PY="${REPO_DIR}/venv/bin/python"
PIP=("$PY" -m pip)
# Suppress "running pip as root" warning when installing into venv (not system)
export PIP_ROOT_USER_ACTION=ignore
echo "  -> venv: $PY"

# --- 4) Dependencies (skip if already satisfied)
echo ""
echo "[4/7] Installing dependencies (skipping if already installed)..."
# Build tools required for SAM2 (bdist_wheel)
"${PIP[@]}" install -q wheel setuptools>=61.0

# PyTorch CUDA 12.1: install or reinstall if broken (e.g. missing cpp_extension / libs)
TORCH_OK=0
if "$PY" -c "import torch; import torch.utils.cpp_extension" 2>/dev/null; then
  TORCH_OK=1
fi
if [ "$TORCH_OK" -eq 0 ]; then
  echo "  Installing PyTorch (CUDA 12.1) and requirements..."
  "${PIP[@]}" uninstall -y torch torchvision torchaudio 2>/dev/null || true
  "${PIP[@]}" install torch torchvision --index-url https://download.pytorch.org/whl/cu121
  "${PIP[@]}" install -r requirements.txt
else
  echo "  Torch OK; ensuring other requirements..."
  "${PIP[@]}" install -q -r requirements.txt
fi

# SAM2 from source (needs torch + wheel; --no-build-isolation uses venv torch)
if ! "$PY" -c "import sam2" 2>/dev/null; then
  echo "  Installing SAM2 from source..."
  export SAM2_BUILD_ALLOW_ERRORS=1
  "${PIP[@]}" install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
fi
echo "  -> Dependencies OK."

# --- 5) Database and paths (persistent inside repo)
echo ""
echo "[5/7] Configuring paths and database..."
mkdir -p data/db
mkdir -p data/raw
mkdir -p outputs

export DATABASE_URL="sqlite:///${REPO_DIR}/data/db/labeling.db"
export RAW_ROOT_DIR="${REPO_DIR}/data/raw"
export OUTPUT_ROOT_DIR="${REPO_DIR}/outputs"
export PS_DEVICE="${PS_DEVICE:-cuda}"
export PS_DTYPE="${PS_DTYPE:-bfloat16}"

echo "  DATABASE_URL=$DATABASE_URL"
echo "  RAW_ROOT_DIR=$RAW_ROOT_DIR"
echo "  OUTPUT_ROOT_DIR=$OUTPUT_ROOT_DIR"
echo "  PS_DEVICE=$PS_DEVICE"
echo "  PS_DTYPE=$PS_DTYPE"
echo "  -> Configuration set (current shell only; add to .env or profile to persist)."

# --- 6) Sanity check
echo ""
echo "[6/7] Sanity check..."
echo "  DATABASE_URL=$DATABASE_URL"
echo "  RAW_ROOT_DIR=$RAW_ROOT_DIR"
echo "  OUTPUT_ROOT_DIR=$OUTPUT_ROOT_DIR"
if "$PY" -c "import torch; print('  CUDA available:', torch.cuda.is_available())"; then
  :
else
  echo "  Note: torch/CUDA check failed (e.g. cupti/driver mismatch). On RunPod with CUDA 12.x it should pass."
fi

# --- 7) Start Gradio UI (or stop after setup)
echo ""
if [ "${1:-}" = "--install-only" ] || [ "${1:-}" = "--setup-only" ]; then
  echo "[7/7] Setup only (--install-only). Skipping Gradio start."
  echo ""
  echo "  To start the UI later, run:"
  echo "    cd $REPO_DIR && source venv/bin/activate"
  echo "    export DATABASE_URL=\"$DATABASE_URL\""
  echo "    export RAW_ROOT_DIR=\"$RAW_ROOT_DIR\""
  echo "    export OUTPUT_ROOT_DIR=\"$OUTPUT_ROOT_DIR\""
  echo "    $PY tools/label_ui_gradio.py"
  echo ""
  exit 0
fi

echo "[7/7] Starting Gradio UI (port 7860, bind 0.0.0.0)..."
echo "  -> Open http://<pod-ip>:7860 or use RunPod's HTTP port mapping."
echo ""
if command -v ss &>/dev/null; then
  ( ss -tulpn 2>/dev/null | grep 7860 ) || true
else
  ( netstat -tulpn 2>/dev/null | grep 7860 ) || true
fi

exec "$PY" tools/label_ui_gradio.py
