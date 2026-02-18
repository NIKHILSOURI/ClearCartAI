#!/usr/bin/env python3
"""
单个商品快速测试脚本
用法: python test_single_product.py <商品编号 1-5>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import ProductSegmentationPipeline
from image_utils import load_images_from_directory

# 5个商品文件夹
PRODUCTS = [
    ("Batteri LR6 AA 63,90 kr", "电池 AA"),
    ("Bad&Toa 6-pack 27,90 kr", "卫生纸 6包"),
    ("Bär Jordgubbar Crunch 25,95", "草莓干"),
    ("Batteri 6 LR61 9V 44,90", "电池 9V"),
    ("Batteri LR03 AAA 64,90", "电池 AAA"),
]

BASE_DIR = Path(r"D:\Download\files (24)")

if len(sys.argv) < 2:
    print("用法: python test_single_product.py <商品编号 1-5>")
    print("\n可用商品:")
    for i, (folder, name) in enumerate(PRODUCTS, 1):
        print(f"  {i}. {name} ({folder})")
    sys.exit(1)

product_idx = int(sys.argv[1]) - 1
if product_idx < 0 or product_idx >= len(PRODUCTS):
    print(f"错误: 商品编号必须在 1-{len(PRODUCTS)} 之间")
    sys.exit(1)

folder_name, product_name = PRODUCTS[product_idx]

print("=" * 80)
print(f"测试商品 #{product_idx + 1}: {product_name}")
print(f"文件夹: {folder_name}")
print("=" * 80)

image_dir = BASE_DIR / folder_name
output_dir = BASE_DIR / f"results_{product_idx + 1}_{folder_name.split()[0]}"

# 加载图片
images = load_images_from_directory(str(image_dir))
image_paths = [p for p, _ in images]

print(f"\n✓ 加载了 {len(image_paths)} 张图片")
print(f"  参考图: {Path(image_paths[0]).name}")

# 初始化 Pipeline
print("\n[初始化] 加载模型...")
pipeline = ProductSegmentationPipeline(
    similarity_threshold=0.60,  # 降低阈值以提高匹配率
    top_k=5,
)

# 运行交互式分割
print("\n[交互式UI] 请在弹出窗口中点击产品...")
print("操作说明：")
print("  • 左键点击：标记产品位置")
print("  • 右键点击：标记背景位置（可选）")
print("  • Enter键：确认选择并开始处理")
print("  • Q键：退出")
print("=" * 80)

result = pipeline.run_interactive(
    image_paths=image_paths,
    reference_image_path=image_paths[0],
    output_dir=str(output_dir),
)

if result:
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    match_rate = f"{result.total_matches}/{result.total_images}"
    percentage = result.total_matches / result.total_images * 100
    print(f"✓ 在 {match_rate} 张图片中找到产品 ({percentage:.1f}%)")
    print(f"✓ 结果保存至: {output_dir}")
    print("\n输出文件：")
    print(f"  • masks/      - 分割掩码")
    print(f"  • cutouts/    - 透明背景抠图")
    print(f"  • overlays/   - 可视化结果")
    print(f"  • gallery.jpg - 结果拼图")
    print(f"  • summary.json - 详细统计")
else:
    print("\n处理已取消或未选择产品。")
