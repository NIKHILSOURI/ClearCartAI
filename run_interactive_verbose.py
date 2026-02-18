#!/usr/bin/env python3
"""
交互式产品分割 - 详细输出版本
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

print("=" * 70)
print("360° 产品分割系统 - 交互式模式")
print("=" * 70)

print("\n[1/5] 加载配置...")
import config
config.print_config()

print("\n[2/5] 加载图片...")
from image_utils import load_images_from_directory

image_dir = r"D:\Download\files (24)\Ägg Eko KRAV M L 50,95"
output_dir = "./results_egg_test"

images = load_images_from_directory(image_dir)
image_paths = [p for p, _ in images]

if not image_paths:
    print("错误：未找到图片！")
    sys.exit(1)

print(f"✓ 成功加载 {len(image_paths)} 张图片")
for i, path in enumerate(image_paths[:3], 1):
    print(f"  {i}. {Path(path).name}")
if len(image_paths) > 3:
    print(f"  ... 还有 {len(image_paths) - 3} 张")

print("\n[3/5] 初始化 Pipeline（首次运行会下载模型，约10GB）...")
print("提示：模型下载可能需要几分钟，请耐心等待...")

from pipeline import ProductSegmentationPipeline

pipeline = ProductSegmentationPipeline(
    similarity_threshold=0.65,
    top_k=3,
)

print("\n[4/5] 启动交互式UI...")
print("=" * 70)
print("操作说明：")
print("  • 左键点击：标记产品位置（正向点）")
print("  • 右键点击：标记背景位置（负向点）")
print("  • Enter键：确认选择")
print("  • R键：重置所有点")
print("  • Q键：退出")
print("=" * 70)

reference = image_paths[0]

result = pipeline.run_interactive(
    image_paths=image_paths,
    reference_image_path=reference,
    output_dir=output_dir,
)

if result:
    print("\n" + "=" * 70)
    print("[5/5] 处理完成！")
    print("=" * 70)
    print(f"✓ 在 {result.total_matches}/{result.total_images} 张图片中找到产品")
    print(f"✓ 结果保存至: {output_dir}")
    print("\n输出文件：")
    print(f"  • masks/      - 分割掩码（PNG）")
    print(f"  • cutouts/    - 透明背景抠图（RGBA）")
    print(f"  • overlays/   - 可视化结果")
    print(f"  • gallery.jpg - 结果拼图")
    print(f"  • summary.json - 详细统计")
else:
    print("\n处理已取消或未选择产品。")
