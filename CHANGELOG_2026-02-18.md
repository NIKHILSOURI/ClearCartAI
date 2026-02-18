# Changelog - 2026-02-18

## Summary

Added cloud-ready labeling infrastructure with PostgreSQL database persistence and Gradio web UI. This enables multiple labelers to work concurrently on product segmentation tasks with proper locking, manual upload capabilities, and database-backed workflow management.

## Features Added

### 1. Database Persistence (PostgreSQL)

- **File**: `db.py`
- **Description**: SQLAlchemy Core-based database module for managing products, images, and labels
- **Tables**:
  - `products`: Product folder metadata
  - `images`: Individual images with labeling status and locking
  - `labels`: Segmentation results with file paths
- **Key Functions**:
  - `init_db()`: Initialize database tables
  - `ingest_dataset()`: Scan and ingest product folders
  - `get_next_unlabeled_image()`: Fetch next image with locking (30min TTL)
  - `save_label()`: Save label and mark image as labeled
  - `healthcheck()`: Database connectivity check

### 2. Manual Upload Feature

- **Location**: `tools/label_ui_gradio.py` (Upload section)
- **Description**: Upload product folders as ZIP files via web UI
- **Process**:
  1. Upload ZIP containing product folder with images
  2. Extract to `RAW_ROOT_DIR` with conflict resolution
  3. Automatically ingest into database
  4. Display upload status and statistics
- **Security**: Zip slip prevention with path validation

### 3. Gradio Labeling UI

- **File**: `tools/label_ui_gradio.py`
- **Description**: Web-based labeling interface integrating existing SAM2 pipeline
- **Features**:
  - Load next unlabeled image from database
  - Interactive point-click segmentation using SAM2
  - Real-time mask visualization
  - Save results to disk + database
  - Automatic load next after save
  - Reset functionality
  - Labeler ID tracking
- **Integration**: Uses existing `ProductSegmentationPipeline` and `SAM2InteractiveSegmenter`

### 4. CLI Ingestion Tool

- **File**: `ingest_folders.py`
- **Description**: Command-line tool for batch ingestion of product folders
- **Usage**:
  ```bash
  python ingest_folders.py --root /data/raw_products --init-db
  ```
- **Features**:
  - Initialize database tables
  - Scan and ingest all product folders
  - Display ingestion summary and statistics

## Files Added

1. **`db.py`** (449 lines)
   - Database module with SQLAlchemy Core
   - Table definitions and CRUD operations
   - Locking mechanism for concurrent labelers

2. **`tools/label_ui_gradio.py`** (436 lines)
   - Gradio web UI for labeling
   - Upload, load, segment, save workflow
   - Database integration

3. **`ingest_folders.py`** (92 lines)
   - CLI tool for dataset ingestion
   - Database initialization option

4. **`docs/CHANGELOG_2026-02-18.md`** (this file)
   - Documentation of changes

## Files Modified

None. All changes are additive to maintain backwards compatibility.

## Configuration

### Environment Variables (Required)

- **`DATABASE_URL`** (required): PostgreSQL connection string
  - Format: `postgresql://user:password@host:port/database`
  - Example: `postgresql://labeler:pass123@localhost:5432/products_db`

### Environment Variables (Optional)

- **`RAW_ROOT_DIR`** (default: `/data/raw_products`): Root directory for raw product images
- **`OUTPUT_ROOT_DIR`** (default: `/data/labels_output`): Root directory for output files (masks/cutouts/overlays)
- **`LOCK_MINUTES`** (default: `30`): Lock timeout in minutes for concurrent labeling

### Directory Structure

```
RAW_ROOT_DIR/
  ├── product_1/
  │   ├── image1.jpg
  │   └── image2.jpg
  └── product_2/
      └── ...

OUTPUT_ROOT_DIR/
  ├── masks/
  │   └── {image_id}.png
  ├── cutouts/
  │   └── {image_id}_white.jpg
  └── overlays/
      └── {image_id}_overlay.jpg
```

## Database Schema

### products
- `id` (PK)
- `name` (text)
- `folder_relpath` (text, unique)
- `created_at` (timestamp)

### images
- `id` (PK)
- `product_id` (FK → products.id)
- `image_relpath` (text, unique)
- `status` (text: 'unlabeled'/'labeled')
- `assigned_to` (text, nullable)
- `locked_at` (timestamp, nullable)
- `created_at` (timestamp)

### labels
- `id` (PK)
- `image_id` (FK → images.id)
- `packaging` (text)
- `product_name` (text)
- `mask_relpath` (text)
- `cutout_relpath` (text)
- `overlay_relpath` (text, nullable)
- `similarity_score` (float, nullable)
- `created_by` (text, nullable)
- `created_at` (timestamp)

## Deployment Instructions

### Local Development

1. **Install dependencies**:
   ```bash
   pip install sqlalchemy psycopg2-binary gradio
   ```

2. **Set up PostgreSQL**:
   ```bash
   # Create database
   createdb products_db
   
   # Set environment variable
   export DATABASE_URL="postgresql://user:pass@localhost:5432/products_db"
   ```

3. **Initialize database**:
   ```bash
   python ingest_folders.py --root ./test_data --init-db
   ```

4. **Launch Gradio UI**:
   ```bash
   python tools/label_ui_gradio.py
   ```
   Access at: http://localhost:7860

### Cloud Deployment

1. **Set environment variables**:
   ```bash
   export DATABASE_URL="postgresql://user:pass@cloud-host:5432/products_db"
   export RAW_ROOT_DIR="/mnt/data/raw_products"
   export OUTPUT_ROOT_DIR="/mnt/data/labels_output"
   export LOCK_MINUTES="30"
   ```

2. **Initialize database** (one-time):
   ```bash
   python ingest_folders.py --root $RAW_ROOT_DIR --init-db
   ```

3. **Deploy Gradio UI**:
   ```bash
   python tools/label_ui_gradio.py
   ```
   Or use Docker/cloud service with port 7860 exposed

### Docker Deployment (Example)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install -r requirements.txt

ENV DATABASE_URL=""
ENV RAW_ROOT_DIR="/data/raw_products"
ENV OUTPUT_ROOT_DIR="/data/labels_output"

EXPOSE 7860

CMD ["python", "tools/label_ui_gradio.py"]
```

## Workflow

### Admin Workflow

1. Upload product folders via Gradio UI (ZIP upload)
2. Or use CLI: `python ingest_folders.py --root /data/raw_products`
3. Monitor database statistics in UI

### Labeler Workflow

1. Enter labeler ID
2. Click "Load Next Image"
3. Click on product to segment (SAM2 generates mask)
4. Fill in packaging type and product name
5. Click "Save & Load Next"
6. Repeat until no more unlabeled images

### Concurrent Labeling

- Images are locked for 30 minutes when assigned
- Expired locks are automatically reclaimed
- No two labelers will get the same image (SQL transaction with `skip_locked`)

## Breaking Changes

None. All existing functionality remains intact.

## Backwards Compatibility

- Existing pipeline code (`pipeline.py`, `sam2_segmenter.py`, etc.) unchanged
- Existing test scripts (`test_single_product.py`, etc.) continue to work
- Database is optional; existing file-based workflow still functional
- `records.jsonl` no longer used for next image selection in Gradio UI

## Dependencies Added

- `sqlalchemy` (≥2.0)
- `psycopg2-binary` (≥2.9)
- `gradio` (≥4.0)

## Known Limitations

1. **Authentication**: No built-in authentication. Deploy behind auth proxy if needed.
2. **File Storage**: Images stored on disk, not in database (by design for performance).
3. **Lock Timeout**: Fixed at 30 minutes (configurable via env var).
4. **Upload Size**: Limited by Gradio/server configuration.

## Future Enhancements

- User authentication and role management
- Batch labeling operations
- Label review and correction workflow
- Export labeled dataset in standard formats (COCO, YOLO, etc.)
- Analytics dashboard
- Multi-language support

## Testing

### Test Database Connection
```bash
python -c "import db; db.init_db(); print('OK' if db.healthcheck() else 'FAIL')"
```

### Test Ingestion
```bash
python ingest_folders.py --root ./test_data --init-db
```

### Test UI
```bash
python tools/label_ui_gradio.py
# Access http://localhost:7860
```

## Support

For issues or questions:
1. Check database connectivity: `DATABASE_URL` is set correctly
2. Verify directories exist: `RAW_ROOT_DIR` and `OUTPUT_ROOT_DIR`
3. Check logs in console output
4. Ensure PostgreSQL is running and accessible

## Contributors

- Initial implementation: 2026-02-18

---

## 2026-02-18 (Update)

### Skip Feature for Low-Quality Images

Added "Skip (Not clear) → Next" action in Gradio labeling UI to handle low-quality or unclear images efficiently.

**Changes:**

1. **Database Module (`db.py`)**:
   - Added `mark_image_skipped()` function to mark images with `status='skipped'`
   - Skipped images are excluded from the default labeling queue
   - Lock is released when image is skipped to prevent blocking other labelers
   - Stores who skipped the image and reason in existing columns

2. **Gradio UI (`tools/label_ui_gradio.py`)**:
   - Added "Skip (Not clear) → Next" button next to Reset button
   - New `skip_image_and_next()` handler function
   - Immediately loads next unlabeled image after skipping
   - No segmentation artifacts (mask/cutout/overlay) are produced for skipped images
   - Updated instructions to mention skip functionality

**Usage:**
- Annotators can click "Skip (Not clear) → Next" for blurry, unclear, or low-quality images
- Skipped images are marked with `status='skipped'` in the database
- The system automatically loads the next unlabeled image
- Skipped images can be reviewed later if needed

**Technical Details:**
- No schema changes required (uses existing `status` column)
- Safe for concurrent labelers (respects existing lock/assignment logic)
- Minimal code changes (~50 lines total)
- Backwards compatible with existing workflow

---

**Version**: 1.0.1  
**Date**: 2026-02-18  
**Status**: Production Ready
