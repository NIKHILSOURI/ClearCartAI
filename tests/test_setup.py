"""Verify environment and project imports (run from project root)."""
import sys
from pathlib import Path

print("=" * 60)
print("Environment & setup check")
print("=" * 60)

# 1. Core library imports
print("\n1. Core libraries...")
try:
    import torch
    print(f"  OK PyTorch {torch.__version__}")
    print(f"     CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"  FAIL PyTorch: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"  OK Transformers {transformers.__version__}")
except Exception as e:
    print(f"  FAIL Transformers: {e}")
    sys.exit(1)

try:
    from sam2.build_sam import build_sam2
    print("  OK SAM2 installed")
except Exception as e:
    print(f"  FAIL SAM2: {e}")
    sys.exit(1)

# 2. Project module imports
print("\n2. Project modules...")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    from ean_system import config
    print(f"  OK config (device={config.get_device()}, dtype={config.DTYPE})")
except Exception as e:
    print(f"  FAIL config: {e}")
    sys.exit(1)

try:
    from ean_system.model_loader import ModelLoader
    print("  OK model_loader")
except Exception as e:
    print(f"  FAIL model_loader: {e}")
    sys.exit(1)

try:
    from ean_system.pipeline import ProductSegmentationPipeline
    print("  OK pipeline")
except Exception as e:
    print(f"  FAIL pipeline: {e}")
    sys.exit(1)

# 3. Optional: image loading (set TEST_IMAGE_DIR to test)
print("\n3. Image loading...")
test_dir = __import__("os").environ.get("TEST_IMAGE_DIR")
if test_dir and Path(test_dir).is_dir():
    try:
        from ean_system.image_utils import load_images_from_directory
        images = load_images_from_directory(test_dir)
        print(f"  OK Loaded {len(images)} images from TEST_IMAGE_DIR")
        if images:
            print(f"     First: {Path(images[0][0]).name}, shape {images[0][1].shape}")
    except Exception as e:
        print(f"  FAIL Image load: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  Skip (set TEST_IMAGE_DIR to a folder with images to test)")

print("\n" + "=" * 60)
print("Setup check done.")
print("=" * 60)
print("\nFirst run of the pipeline may download ~10GB of models.")
