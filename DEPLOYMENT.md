# Deployment Guide - EAN Vision System

## 📦 What to Deliver to Others

### ✅ **INCLUDE These Files/Folders**

```
ean_system/
├── README.md                    ✅ Main documentation
├── CHANGELOG.md                 ✅ Change history
├── DEPLOYMENT.md                ✅ This file
├── requirements.txt             ✅ Python dependencies
├── setup.ps1                    ✅ Windows setup script
├── setup.sh                     ✅ Linux/Mac setup script
├── quick_start.ps1              ✅ Quick launcher (Windows)
├── start_labeling_fixed.py      ✅ Alternative launcher
├── .gitignore                   ✅ Git configuration
├── configs/                     ✅ Configuration files
│   └── system.yaml
├── src/                         ✅ Source code
│   └── ean/
├── tools/                       ✅ Labeling tool
│   └── label_ui_gradio.py
├── scripts/                     ✅ Utility scripts
└── sam2_b.pt                    ✅ SAM model weights (161 MB)
```

### ❌ **DO NOT INCLUDE**

```
❌ .venv/                        - Virtual environment (recipient creates their own)
❌ raw_pictures/                 - User's images (unless sharing dataset)
❌ data/labels/                  - User's labels (unless sharing dataset)
❌ models/                       - Trained models (unless sharing trained models)
❌ flagged/                      - Gradio debug folder
❌ __pycache__/                  - Python cache
❌ *.pyc                         - Compiled Python files
```

---

## 📋 Delivery Checklist

### Option 1: Clean Code Delivery (Recommended)
**What**: Source code only, recipient sets up their own environment

**Include**:
- ✅ All source code (`src/`, `tools/`, `scripts/`)
- ✅ Configuration files (`configs/`)
- ✅ Setup scripts (`setup.ps1`, `setup.sh`)
- ✅ Documentation (`README.md`, `CHANGELOG.md`, `DEPLOYMENT.md`)
- ✅ Dependencies list (`requirements.txt`)
- ✅ SAM model weights (`sam2_b.pt`)

**Exclude**:
- ❌ `.venv/` (recipient creates their own)
- ❌ `data/` (recipient generates their own)
- ❌ `models/` (recipient trains their own)

**Size**: ~200 MB (mostly SAM model)

---

### Option 2: Full Package with Trained Models
**What**: Everything including trained models

**Include**: Everything from Option 1, PLUS:
- ✅ `models/packaging_router/` (trained packaging classifier)
- ✅ `models/product_models/` (trained product classifiers)
- ✅ `data/mappings/product_to_ean.csv` (EAN mapping)

**Exclude**:
- ❌ `.venv/` (still exclude)
- ❌ `raw_pictures/` (unless sharing dataset)
- ❌ `data/labels/` (unless sharing dataset)

**Size**: ~500 MB - 2 GB (depending on trained models)

---

### Option 3: Complete Dataset Package
**What**: Everything including labeled data

**Include**: Everything from Option 2, PLUS:
- ✅ `raw_pictures/` (original images)
- ✅ `data/labels/records.jsonl` (label records)
- ✅ `data/labels/masks/` (segmentation masks)
- ✅ `data/labels/crops/` (cropped products)
- ✅ `data/datasets/` (training datasets)

**Exclude**:
- ❌ `.venv/` (always exclude)

**Size**: 5-50 GB (depending on image count)

---

## 🚀 Recipient Setup Instructions

### Step 1: Extract Package
```powershell
# Windows
Expand-Archive ean_system.zip -DestinationPath C:\Projects\

# Linux/Mac
unzip ean_system.zip -d ~/projects/
```

### Step 2: Run Setup Script
```powershell
# Windows
cd ean_system
.\setup.ps1

# Linux/Mac
cd ean_system
chmod +x setup.sh
./setup.sh
```

**What setup.ps1 does**:
1. Creates Python virtual environment (`.venv/`)
2. Installs all dependencies from `requirements.txt`
3. Verifies SAM model exists
4. Creates necessary directories

### Step 3: Add Images
```powershell
# Copy product images to raw_pictures folder
Copy-Item "C:\MyImages\*.jpg" -Destination "raw_pictures\"
```

### Step 4: Start Labeling Tool
```powershell
# Quick start (recommended)
.\quick_start.ps1

# Or manual start
$env:PYTHONPATH = "C:\Projects\ean_system"
.\.venv\Scripts\python.exe tools\label_ui_gradio.py
```

### Step 5: Open Browser
```
http://127.0.0.1:7860
```

---

## 📝 Important Notes for Recipients

### System Requirements
- **OS**: Windows 10/11, Linux, or macOS
- **Python**: 3.10 or 3.11 (3.12+ not tested)
- **RAM**: 8 GB minimum, 16 GB recommended
- **GPU**: Optional but recommended for faster SAM inference
- **Disk**: 10 GB free space (more if storing many images)

### Network Requirements
- **VPN**: Must be disabled when running labeling tool
- **Firewall**: Allow localhost connections on port 7860
- **Internet**: Required during setup (to download dependencies)

### Dependencies
All Python packages are listed in `requirements.txt`:
- `gradio==3.50.2` (UI framework)
- `ultralytics` (YOLO models)
- `opencv-python` (image processing)
- `torch` (PyTorch for SAM)
- And others...

---

## 🔧 Troubleshooting for Recipients

### Issue: "Virtual environment not found"
**Solution**: Run `setup.ps1` first

### Issue: "No images found"
**Solution**: Add .jpg images to `raw_pictures/` folder

### Issue: "Port 7860 already in use"
**Solution**: 
```powershell
# Find and kill process using port 7860
Get-Process | Where-Object {$_.ProcessName -eq "python"} | Stop-Process -Force
```

### Issue: "WinError 10054" or network errors
**Solution**: Disable VPN and restart tool

### Issue: "SAM model not found"
**Solution**: Ensure `sam2_b.pt` (161 MB) is in root directory

---

## 📦 How to Package for Delivery

### Method 1: ZIP Archive (Recommended)
```powershell
# Windows - Create clean package
$exclude = @('.venv', 'data', 'models', 'flagged', '__pycache__', '*.pyc')
Compress-Archive -Path * -DestinationPath ..\ean_system_v1.0.zip -Force

# Linux/Mac
zip -r ../ean_system_v1.0.zip . -x ".venv/*" "data/*" "models/*" "flagged/*" "*/__pycache__/*" "*.pyc"
```

### Method 2: Git Repository
```bash
# Initialize git repo (if not already)
git init
git add .
git commit -m "Initial commit - EAN Vision System"

# Push to GitHub/GitLab
git remote add origin https://github.com/yourusername/ean_system.git
git push -u origin main
```

**Note**: `.gitignore` already excludes `.venv/`, `data/`, etc.

---

## 📄 Documentation to Include

### README.md
- Project overview
- Quick start guide
- System architecture
- API documentation

### CHANGELOG.md
- Recent improvements
- Bug fixes
- Feature changes
- Version history

### DEPLOYMENT.md (this file)
- Delivery instructions
- Setup guide
- Troubleshooting

---

## ✅ Pre-Delivery Checklist

Before sending to recipient:

- [ ] Remove `.venv/` folder
- [ ] Remove `data/labels/` (unless sharing dataset)
- [ ] Remove `models/` (unless sharing trained models)
- [ ] Remove `flagged/` folder
- [ ] Remove all `__pycache__/` folders
- [ ] Verify `sam2_b.pt` is included
- [ ] Verify `requirements.txt` is up to date
- [ ] Test `setup.ps1` on clean machine
- [ ] Test `quick_start.ps1` after setup
- [ ] Update README.md with any final notes
- [ ] Create ZIP or push to Git repository

---

## 🎯 Summary

**For most cases, use Option 1 (Clean Code Delivery)**:
- Small package size (~200 MB)
- Recipient creates their own environment
- No unnecessary files
- Easy to version control

**Package contents**:
```
ean_system_v1.0.zip
├── Source code
├── Configuration
├── Documentation
├── Setup scripts
└── SAM model weights
```

**Recipient runs**:
1. `setup.ps1` (creates environment)
2. Add images to `raw_pictures/`
3. `quick_start.ps1` (starts tool)
4. Open browser to `http://127.0.0.1:7860`

**That's it!** 🚀
