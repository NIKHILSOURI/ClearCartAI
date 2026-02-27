# ============================================================
# EAN Product Segmentation - Install and Run Labeling UI
# Windows (PowerShell). Run from project root: .\install_and_run_ui.ps1
# ============================================================

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

Set-Location $ProjectRoot

Write-Host "============================================"
Write-Host " EAN Product Segmentation - Install & Run UI"
Write-Host "============================================"
Write-Host ""

# --- 1. Virtual environment ---
if (-not (Test-Path "venv")) {
    Write-Host "[1/5] Creating virtual environment..."
    python -m venv venv
} else {
    Write-Host "[1/5] Using existing venv"
}

$pip = Join-Path $ProjectRoot "venv\Scripts\pip.exe"
$python = Join-Path $ProjectRoot "venv\Scripts\python.exe"

if (-not (Test-Path $pip)) {
    Write-Host "ERROR: venv not found or broken. Remove 'venv' and run again." -ForegroundColor Red
    exit 1
}

# --- 2. Dependencies ---
Write-Host "[2/5] Installing dependencies from requirements.txt..."
& $pip install -q -r requirements.txt

# --- 3. PyTorch with CUDA ---
Write-Host "[3/5] Installing PyTorch with CUDA 12.1..."
& $pip uninstall -y torch torchvision torchaudio 2>$null
& $pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cu121

# --- 4. SAM2 ---
Write-Host "[4/5] Installing SAM2 from source..."
& $pip install -q "git+https://github.com/facebookresearch/sam2.git"

# --- 5. Verify CUDA ---
Write-Host "[5/5] Verifying CUDA..."
& $python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

Write-Host ""
Write-Host "============================================"
Write-Host " Install complete!"
Write-Host "============================================"
Write-Host ""

# --- Run UI ---
$runUi = $true
if ($args -contains "--install-only") {
    $runUi = $false
    Write-Host "Install only (--install-only). To run the UI later:"
    Write-Host "  venv\Scripts\activate"
    Write-Host "  `$env:DATABASE_URL = 'postgresql://user:pass@localhost:5432/dbname'"
    Write-Host "  python tools/label_ui_gradio.py"
    Write-Host ""
    exit 0
}

if (-not $env:DATABASE_URL) {
    Write-Host "WARNING: DATABASE_URL is not set. Set it for the labeling UI, e.g.:" -ForegroundColor Yellow
    Write-Host "  `$env:DATABASE_URL = 'postgresql://user:password@localhost:5432/ean_labels'" -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/N)"
    if ($continue -ne "y" -and $continue -ne "Y") {
        Write-Host "Set DATABASE_URL and run: .\venv\Scripts\python.exe tools/label_ui_gradio.py"
        exit 0
    }
}

Write-Host "Starting Gradio labeling UI on http://localhost:7860 ..."
Write-Host "Press Ctrl+C to stop."
Write-Host ""
& $python tools/label_ui_gradio.py
