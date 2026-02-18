#!/usr/bin/env python3
"""
批量交互式测试 - 先收集所有点击，再统一处理
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import ProductSegmentationPipeline
from image_utils import load_images_from_directory
from interactive_selector import InteractiveSelector
import numpy as np

# 6个商品文件夹
PRODUCTS = [
    ("Ägg Eko KRAV M L 50,95", "鸡蛋盒"),
    ("Bad&Toa 6-pack 27,90 kr", "卫生纸 6包"),
    ("Bär Jordgubbar Crunch 25,95", "草莓干"),
    ("Batteri LR03 AAA 64,90", "电池 AAA"),
    ("Batteri LR6 AA 63,90 kr", "电池 AA"),
]

BASE_DIR = Path(r"D:\Download\files (24)")

print("=" * 80)
print("批量交互式产品分割测试 - 6个商品")
print("=" * 80)
print("\n第一阶段：依次为每个商品点击选择产品位置")
print("第二阶段：系统自动批量处理所有商品")
print("=" * 80)

# 初始化 Pipeline（只需要一次）
print("\n[初始化] 加载模型...")
pipeline = ProductSegmentationPipeline(
    similarity_threshold=0.60,
    top_k=5,
)
print("✓ 模型加载完成\n")

# 第一阶段：收集所有商品的点击
collected_data = []

for idx, (folder_name, product_name) in enumerate(PRODUCTS, 1):
    print("=" * 80)
    print(f"[{idx}/{len(PRODUCTS)}] 商品: {product_name}")
    print(f"文件夹: {folder_name}")
    print("=" * 80)
    
    image_dir = BASE_DIR / folder_name
    output_dir = BASE_DIR / f"results_{idx}_{folder_name.split()[0]}"
    
    # 加载图片
    images = load_images_from_directory(str(image_dir))
    image_paths = [p for p, _ in images]
    image_arrays = [img for _, img in images]
    
    print(f"✓ 加载了 {len(image_paths)} 张图片")
    print(f"  参考图: {Path(image_paths[0]).name}")
    
    # 交互式选择产品
    print(f"\n[{idx}/{len(PRODUCTS)}] 请在弹出窗口中点击产品...")
    print("操作说明：")
    print("  • 左键点击：标记产品位置")
    print("  • 右键点击：标记背景位置（可选）")
    print("  • Enter键：确认选择，进入下一个商品")
    print("  • Q键：跳过此商品")
    print("-" * 80)
    
    try:
        # 确保模型已加载
        pipeline._ensure_models()
        
        # 使用交互式选择器
        selector = InteractiveSelector(segmenter=pipeline.interactive_segmenter)
        mask, bbox = selector.select_product(
            image=image_arrays[0],
            image_path=image_paths[0],
        )
        
        if mask is not None and bbox is not None:
            print(f"✓ 收集到产品选择")
            collected_data.append({
                "product_name": product_name,
                "folder_name": folder_name,
                "image_paths": image_paths,
                "reference_path": image_paths[0],
                "reference_image": image_arrays[0],
                "mask": mask,
                "bbox": bbox,
                "output_dir": str(output_dir),
            })
        else:
            print(f"⊘ 跳过商品: {product_name}")
    
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
    
    print()

# 第二阶段：批量处理所有收集到的商品
if not collected_data:
    print("\n没有收集到任何商品数据，退出。")
    sys.exit(0)

print("\n" + "=" * 80)
print(f"第二阶段：开始批量处理 {len(collected_data)} 个商品")
print("=" * 80)

results_summary = []

for idx, data in enumerate(collected_data, 1):
    print("\n" + "=" * 80)
    print(f"[{idx}/{len(collected_data)}] 处理: {data['product_name']}")
    print("=" * 80)
    
    try:
        # 使用收集的 bbox 数据运行分割
        result = pipeline.run_with_bbox(
            image_paths=data['image_paths'],
            reference_image_path=data['reference_path'],
            reference_bbox=data['bbox'],
            output_dir=data['output_dir'],
        )
        
        if result:
            match_rate = f"{result.total_matches}/{result.total_images}"
            percentage = result.total_matches / result.total_images * 100
            print(f"✓ 完成！在 {match_rate} 张图片中找到产品 ({percentage:.1f}%)")
            print(f"✓ 结果保存至: {data['output_dir']}")
            
            results_summary.append({
                "product": data['product_name'],
                "folder": data['folder_name'],
                "matches": result.total_matches,
                "total": result.total_images,
                "rate": f"{percentage:.1f}%",
                "output": data['output_dir'],
            })
        else:
            print(f"⊘ 处理失败")
            results_summary.append({
                "product": data['product_name'],
                "folder": data['folder_name'],
                "matches": 0,
                "total": len(data['image_paths']),
                "rate": "0%",
                "output": "处理失败",
            })
    
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        results_summary.append({
            "product": data['product_name'],
            "folder": data['folder_name'],
            "matches": 0,
            "total": len(data['image_paths']),
            "rate": "错误",
            "output": str(e),
        })

# 打印汇总结果
print("\n" + "=" * 80)
print("批量测试完成 - 结果汇总")
print("=" * 80)
print(f"\n{'商品':<20} {'匹配率':<15} {'结果目录':<10}")
print("-" * 80)

for r in results_summary:
    status = "✓" if r["matches"] > 0 else "⊘"
    print(f"{status} {r['product']:<18} {r['matches']}/{r['total']} ({r['rate']:<6})")

print("\n" + "=" * 80)
success_count = len([r for r in results_summary if r['matches'] > 0])
print(f"总计: {success_count}/{len(results_summary)} 个商品成功处理")
print("=" * 80)

# 打印详细结果路径
print("\n详细结果路径：")
for r in results_summary:
    if r['matches'] > 0:
        print(f"  • {r['product']}: {r['output']}")
