# Changelog - EAN Vision System Labeling Tool

## Recent Improvements and Bug Fixes

### 🎯 Core Features Enhanced

#### 1. **Sequential Image Browsing**
- **Problem**: Load Next button always returned the same first unlabeled image
- **Solution**: Implemented global image list with index tracking
- **Result**: Users can now continuously browse through all 206 images in sequence
- **Behavior**: 
  - Click Load Next repeatedly to cycle through all images
  - No need to save to move to next image
  - Shows progress: `[current/total]` and label status
  - Loops back to first image after reaching the end

#### 2. **Race Condition Prevention (UID Tracking)**
- **Problem**: SAM segmentation results from old image would overwrite new image when switching during inference
- **Solution**: Added unique identifier (UID) to each loaded image state
- **Implementation**:
  - Each image gets `uid: time.time()` when loaded
  - `run_sam()` captures UID at start, validates at end
  - If UID changed (user switched images), result is discarded
- **Result**: Status shows `⚠️ Image changed during SAM run. Result discarded. Please run again.`

#### 3. **Concurrency Control**
- **Problem**: Multiple button clicks could trigger parallel operations
- **Solution**: Set `demo.queue(concurrency_count=1)` for Gradio 3.x
- **Result**: All operations execute serially, preventing state corruption

#### 4. **Error Handling & User Feedback**
- **Added**: Try/except blocks in all critical functions:
  - `load_next()`
  - `run_sam()`
  - `save_record_and_next()`
- **Added**: Status messages with `.then()` chaining:
  - `⏳ Running SAM segmentation... Please wait (8-10 seconds)...`
  - `⏳ Saving record and loading next image...`
  - `✅ Loaded: [filename]`
  - `❌ Error: [details]`

#### 5. **State Management**
- **Fixed**: `gr.State` initialization for Gradio 3.x compatibility
- **Changed**: `gr.State({})` → `gr.State(value=None)`
- **Added**: Proper state reset on image load:
  - `points: []`
  - `labels: []`
  - `last_mask: None`
  - `last_crop_bgr: None`

### 🐛 Bug Fixes

#### Gradio Version Compatibility
- **Issue**: Gradio 4.x had schema inference bugs causing `TypeError: argument of type 'bool' is not iterable`
- **Solution**: Downgraded to stable Gradio 3.50.2
- **Dependencies**:
  - `gradio==3.50.2`
  - `gradio-client==0.6.1`
  - `huggingface-hub==1.4.1`

#### VPN Network Interference
- **Issue**: VPN blocked localhost connections causing `WinError 10054`
- **Solution**: Documented requirement to disable VPN when running labeling tool
- **Root Cause**: Gradio's internal health check (`url_ok()`) failed with VPN active

#### Image Component Type Parameter
- **Issue**: `gr.Image(type="pil")` caused compatibility issues
- **Solution**: Removed `type` parameter, let Gradio auto-infer

### 🔧 Technical Changes

#### Code Structure
```python
# Global image tracking
_all_images = None
_current_index = -1

def _get_all_images():
    """Cache and return sorted list of all images"""
    global _all_images
    if _all_images is None:
        _all_images = sorted(list_images(cfg.raw_dir))
    return _all_images

def _next_image() -> Optional[Path]:
    """Get next image in sequence, cycling through all images"""
    global _current_index
    all_imgs = _get_all_images()
    _current_index = (_current_index + 1) % len(all_imgs)
    return all_imgs[_current_index]
```

#### Launch Configuration
```python
# Simplified for Gradio 3.x
demo.queue(concurrency_count=1)
demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
    share=False,
    inbrowser=False,
    show_error=True
)
```

### 📊 Current Status

| Feature | Status |
|---------|--------|
| Sequential image browsing | ✅ Working |
| Load Next without saving | ✅ Working |
| UID-based race condition prevention | ✅ Working |
| SAM segmentation | ✅ Working |
| Error feedback | ✅ Working |
| Progress tracking | ✅ Working |
| Concurrency control | ✅ Working |

### 🚀 Usage

1. **Ensure VPN is disabled**
2. Start server: 
   ```powershell
   $env:PYTHONPATH="D:\Download\ean_system_repo\ean_system"
   .\.venv\Scripts\python.exe tools\label_ui_gradio.py
   ```
3. Open browser: `http://127.0.0.1:7860`
4. Click "Load Next" to browse images
5. Click product center → "Run SAM 3" → Fill info → "Save Record"

### 📝 Notes

- Total images: 206
- Currently labeled: 8
- Remaining: 198
- Image directory: `raw_pictures/`
- Records saved to: `data/labels/records.jsonl`
- Masks saved to: `data/labels/masks/`
- Crops saved to: `data/labels/crops/`
