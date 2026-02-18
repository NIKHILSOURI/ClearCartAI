"""测试环境设置和模型加载"""
import sys
from pathlib import Path

print("=" * 60)
print("测试环境设置")
print("=" * 60)

# 测试基本导入
print("\n1. 测试基本库导入...")
try:
    import torch
    print(f"✓ PyTorch {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA device: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"✗ PyTorch 导入失败: {e}")
    sys.exit(1)

try:
    import transformers
    print(f"✓ Transformers {transformers.__version__}")
except Exception as e:
    print(f"✗ Transformers 导入失败: {e}")
    sys.exit(1)

try:
    from sam2.build_sam import build_sam2
    print(f"✓ SAM2 已安装")
except Exception as e:
    print(f"✗ SAM2 导入失败: {e}")
    sys.exit(1)

# 测试项目模块导入
print("\n2. 测试项目模块导入...")
sys.path.insert(0, str(Path(__file__).parent))

try:
    import config
    print(f"✓ config 模块")
    print(f"  Device: {config.get_device()}")
    print(f"  DType: {config.DTYPE}")
except Exception as e:
    print(f"✗ config 导入失败: {e}")
    sys.exit(1)

try:
    from model_loader import ModelLoader
    print(f"✓ model_loader 模块")
except Exception as e:
    print(f"✗ model_loader 导入失败: {e}")
    sys.exit(1)

try:
    from pipeline import ProductSegmentationPipeline
    print(f"✓ pipeline 模块")
except Exception as e:
    print(f"✗ pipeline 导入失败: {e}")
    sys.exit(1)

# 测试图片加载
print("\n3. 测试图片加载...")
try:
    from image_utils import load_images_from_directory
    test_dir = r"D:\Download\files (24)\Ägg Eko KRAV M L 50,95"
    images = load_images_from_directory(test_dir)
    print(f"✓ 成功加载 {len(images)} 张图片")
    if images:
        print(f"  第一张图片: {Path(images[0][0]).name}")
        print(f"  图片尺寸: {images[0][1].shape}")
except Exception as e:
    print(f"✗ 图片加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("环境测试完成！")
print("=" * 60)
print("\n提示：首次运行会自动下载模型（约10GB），请耐心等待...")
