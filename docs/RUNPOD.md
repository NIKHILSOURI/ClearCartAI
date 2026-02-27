## RunPod Deployment Guide

For a **single guide** that covers **running locally**, **deploying locally**, and **RunPod**, see **[LOCAL_AND_RUNPOD.md](LOCAL_AND_RUNPOD.md)**.

The rest of this file is a **RunPod-only** reference.

---

This document explains how to deploy the `ean_system_new` product segmentation and labeling UI to **RunPod** for hourly GPU usage.

It focuses on:
- **GPU + system requirements**
- **Recommended RunPod template**
- **Environment variables and storage**
- **Commands to start the web UI**

---

## 1. What runs on RunPod?

Two main entry points are relevant for hosting:

- **Labeling UI (Gradio web app)**  
  - File: `tools/label_ui_gradio.py`  
  - Purpose: product labeling with SAM2 + DINOv2, backed by PostgreSQL  
  - Port: **7860** (binds to `0.0.0.0:7860`)

- **Batch / interactive scripts (optional for debugging)**  
  - `scripts/run_interactive.py`, `scripts/run_batch.py`, etc.  
  - These can be run from the same pod for one‑off jobs, but are not required for the labeling UI.

For a typical deployment, you run **only**:

```bash
python tools/label_ui_gradio.py
```

and expose port `7860` in RunPod.

---

## 2. Hardware requirements (RunPod)

The system uses:

- **SAM2** – `facebook/sam2.1-hiera-large`
- **DINOv2** – `facebook/dinov2-large`
- **PyTorch with CUDA** + automatic mixed precision (`bfloat16` by default)

These are fairly heavy models.

**Recommended minimum for smooth operation:**

- **GPU VRAM**: **≥ 16 GB**
  - Example RunPod GPUs: **T4 16GB**, **L4 24GB**, **A10G 24GB**, etc.
- **System RAM**: **≥ 16 GB**
- **Disk**: **≥ 30 GB**
  - ~10+ GB for SAM2 + DINOv2 checkpoints from Hugging Face
  - Remaining for Python environment, logs, and output images
- **Python**: 3.10 or newer
- **CUDA / Drivers**: compatible with PyTorch CUDA wheels (e.g. `cu121`)

You may be able to run on **12 GB VRAM** with small images and careful usage, but for reliability the target is **16 GB+**, ideally **24 GB**.

---

## 3. Choosing a RunPod template

On RunPod:

1. Create a **Secure Cloud Compute** pod.
2. Select a **PyTorch + CUDA** template image. Any recent CUDA 12.x PyTorch image is fine.
3. Choose a GPU with **≥ 16 GB VRAM** (or 24 GB for extra headroom).
4. Allocate:
   - **vCPUs**: 4+ (more improves responsiveness)
   - **RAM**: 16–32 GB
   - **Disk**: 50+ GB (to be safe with models and data)

---

## 4. Directory layout inside the pod

Assume you clone the repo under `/workspace`:

```bash
cd /workspace
git clone <YOUR_REPO_URL>
cd CLEARCART/Ean-System/ean_system_new
```

Project root here is `ean_system_new/` (this matches `config.PROJECT_ROOT`).

---

## 5. Environment variables

Set the following **RunPod environment variables** (through the pod UI or startup script):

### 5.1 Database (required for labeling UI)

- **`DATABASE_URL`** – PostgreSQL connection string. Replace the placeholders with your real DB host, user, password, and database name:

```text
postgresql://user:password@db-hostname:5432/dbname
```

**Where to set it on RunPod:**

1. **RunPod dashboard** – When creating or editing the pod: **Pod** → **Edit** (or **Deploy**), then find **Environment Variables** and add:
   - **Key:** `DATABASE_URL`
   - **Value:** `postgresql://your_user:your_password@your_db_host:5432/your_dbname`

2. **Inside the pod (shell or startup script)** – Before starting the app, export the variable:
   ```bash
   export DATABASE_URL="postgresql://your_user:your_password@your_db_host:5432/your_dbname"
   python tools/label_ui_gradio.py
   ```

Use your actual PostgreSQL host (e.g. a managed DB URL, or another RunPod pod’s IP if Postgres runs there), username, password, and database name. The app reads `DATABASE_URL` at startup and uses it to check DB health, ingest folders, and store labels.

### 5.2 Paths for images and outputs

Defaults in `tools/label_ui_gradio.py`:

- **`RAW_ROOT_DIR`** (default `/data/raw_products`)
- **`OUTPUT_ROOT_DIR`** (default `/data/labels_output`)
- **`LOCK_MINUTES`** (default `30`)

In RunPod:

- Mount a persistent volume to `/data` (or your preferred path).
- Optional: override the defaults, for example:

```text
RAW_ROOT_DIR=/data/raw_products
OUTPUT_ROOT_DIR=/data/labels_output
LOCK_MINUTES=30
```

### 5.3 Model configuration (optional overrides)

Defaults in `ean_system/config.py`:

- `PS_SAM2_CHECKPOINT` (default `sam2.1_hiera_large.pt`)
- `PS_SAM2_CONFIG` (default `configs/sam2.1/sam2.1_hiera_l.yaml`)
- `PS_DINOV2_MODEL` (default `facebook/dinov2-large`)
- `PS_SIMILARITY_THRESHOLD` (default `0.65`)
- `PS_DEVICE` (default `"cuda"` if `CUDA_VISIBLE_DEVICES` is set, else `"auto"`)
- `PS_DTYPE` (default `"bfloat16"`)

You can override any of them using environment variables. Examples:

```text
PS_DEVICE=cuda
PS_DTYPE=bfloat16
PS_SIMILARITY_THRESHOLD=0.7
```

---

## 6. Installing dependencies inside the pod

From the `ean_system_new` directory:

```bash
cd /workspace/CLEARCART/Ean-System/ean_system_new

# Optional but recommended: virtual environment
python -m venv venv
source venv/bin/activate  # Windows pods: venv\Scripts\activate

# 1. Core Python dependencies
pip install -r requirements.txt

# 2. Install CUDA-enabled PyTorch (adjust cu121 to pod image CUDA version if needed)
pip uninstall -y torch torchvision torchaudio || true
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 3. Install SAM2 from source (used by ModelLoader and sam2_segmenter)
pip install git+https://github.com/facebookresearch/sam2.git
```

On first use, SAM2 and DINOv2 will be downloaded from Hugging Face (about ~10 GB total).

You can pre‑warm this by running a small test, for example:

```bash
python tests/test_setup.py
```

---

## 7. Starting the Gradio labeling UI

After dependencies are installed and environment variables are set:

```bash
cd /workspace/CLEARCART/Ean-System/ean_system_new
source venv/bin/activate

python tools/label_ui_gradio.py
```

This will:

- Initialize the PostgreSQL database (via `ean_system.db`)
- Start a **Gradio** app bound to `0.0.0.0:7860`

### 7.1 Exposing the UI on RunPod

In the pod settings:

- Expose **TCP port 7860**
- Use RunPod’s generated HTTP endpoint to access the Gradio interface from your browser

---

## 8. Optional: Running pipeline scripts on the same pod

You can also run the core segmentation pipeline scripts on the same GPU pod for manual or batch runs.

From the project root (`ean_system_new/`):

```bash
# Interactive matplotlib-based UI (local only)
python scripts/run_interactive.py --image-dir ./path/to/images --output ./outputs

# Pure batch: reference point
python scripts/run_batch.py \
  --image-dir ./path/to/images \
  --reference img_001.jpg \
  --point 250 300 \
  --output ./outputs
```

These use the same models and configuration as the labeling UI, so the **same GPU and environment** are sufficient.

---

## 9. Quick checklist for RunPod

Before starting:

- **GPU**: ≥ 16 GB VRAM (preferably 24 GB)
- **RAM**: ≥ 16 GB
- **Disk**: ≥ 30–50 GB
- **Environment**:
  - `DATABASE_URL` set and reachable
  - `RAW_ROOT_DIR` and `OUTPUT_ROOT_DIR` (and volume mounted)
  - Optional `PS_*` overrides as needed
- **Commands**:
  - Install deps (`requirements.txt`, PyTorch CUDA, SAM2)
  - Run `python tools/label_ui_gradio.py`
  - Expose port `7860` in RunPod

Once running, you can:

- Upload ZIPs of product images via the **Upload Product Folder** section
- Label products interactively using SAM2 masks
- Store labels in PostgreSQL and on the mounted volume

