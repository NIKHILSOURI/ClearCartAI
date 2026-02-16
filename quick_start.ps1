# EAN Vision System - Quick Start Script
# Launches the labeling tool with one command

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  EAN Vision System - Labeling Tool" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Please run setup.ps1 first to create the environment." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Run: .\setup.ps1" -ForegroundColor Green
    Write-Host ""
    exit 1
}

# Check if images directory exists
if (-not (Test-Path "raw_pictures")) {
    Write-Host "WARNING: raw_pictures directory not found!" -ForegroundColor Yellow
    Write-Host "Creating raw_pictures directory..." -ForegroundColor Yellow
    New-Item -ItemType Directory -Path "raw_pictures" -Force | Out-Null
    Write-Host "Please add your product images to the raw_pictures folder." -ForegroundColor Green
    Write-Host ""
}

# Check if there are images
$imageCount = (Get-ChildItem raw_pictures\*.jpg -ErrorAction SilentlyContinue | Measure-Object).Count
if ($imageCount -eq 0) {
    Write-Host "WARNING: No images found in raw_pictures folder!" -ForegroundColor Yellow
    Write-Host "Please add .jpg images to raw_pictures/ before starting." -ForegroundColor Yellow
    Write-Host ""
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 0
    }
} else {
    Write-Host "Found $imageCount images in raw_pictures/" -ForegroundColor Green
}

# Check labeled records
if (Test-Path "data\labels\records.jsonl") {
    $labeledCount = (Get-Content data\labels\records.jsonl -ErrorAction SilentlyContinue | Measure-Object -Line).Lines
    Write-Host "Already labeled: $labeledCount images" -ForegroundColor Green
} else {
    Write-Host "No previous labels found. Starting fresh." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Labeling Tool..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Server will start at: http://127.0.0.1:7860" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT:" -ForegroundColor Yellow
Write-Host "  - Disable VPN before using the tool" -ForegroundColor Yellow
Write-Host "  - Click 'Load Next' to browse images" -ForegroundColor Yellow
Write-Host "  - SAM processing takes 8-10 seconds" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop the server" -ForegroundColor Gray
Write-Host ""

# Set Python path
$env:PYTHONPATH = $PSScriptRoot

# Start the labeling tool
& .\.venv\Scripts\python.exe tools\label_ui_gradio.py
