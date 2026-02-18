#!/usr/bin/env python3
"""
批量测试5个商品的产品分割
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

print("=" * 80)
print("批量产品分割测试 - 5个商品")
print("=" * 80)

# 初始化 Pipeline（只需要一次）
print("\n[初始化] 加载模型...")
pipeline = ProductSegmentationPipeline(
    similarity_threshold=0.60,  # 降低阈值以提高匹配率
    top_k=5,
)
print("✓ 模型加载完成\n")

results_summary = []

for idx, (folder_name, product_name) in enumerate(PRODUCTS, 1):
    print("=" * 80)
    print(f"[{idx}/5] 处理商品: {product_name}")
    print(f"文件夹: {folder_name}")
    print("=" * 80)
    
    image_dir = BASE_DIR / folder_name
    output_dir = BASE_DIR / f"results_{idx}_{folder_name.split()[0]}"
    
    # 加载图片
    images = load_images_from_directory(str(image_dir))
    image_paths = [p for p, _ in images]
    
    print(f"✓ 加载了 {len(image_paths)} 张图片")
    print(f"  参考图: {Path(image_paths[0]).name}")
    
    # 运行交互式分割
    print("\n[交互式UI] 请在弹出窗口中点击产品...")
    print("操作说明：")
    print("  • 左键点击：标记产品位置")
    print("  • 右键点击：标记背景位置（可选）")
    print("  • Enter键：确认选择并开始处理")
    print("  • Q键：跳过此商品")
    print("-" * 80)
    
    try:
        result = pipeline.run_interactive(
            image_paths=image_paths,
            reference_image_path=image_paths[0],
            output_dir=str(output_dir),
        )
        
        if result:
            match_rate = f"{result.total_matches}/{result.total_images}"
            print(f"\n✓ 完成！在 {match_rate} 张图片中找到产品")
            print(f"✓ 结果保存至: {output_dir}")
            results_summary.append({
                "product": product_name,
                "folder": folder_name,
                "matches": result.total_matches,
                "total": result.total_images,
                "rate": f"{result.total_matches/result.total_images*100:.1f}%",
                "output": str(output_dir),
            })
        else:
            print(f"\n⊘ 跳过或取消")
            results_summary.append({
                "product": product_name,
                "folder": folder_name,
                "matches": 0,
                "total": len(image_paths),
                "rate": "0%",
                "output": "未处理",
            })
    except Exception as e:
        print(f"\n✗ 错误: {e}")
        results_summary.append({
            "product": product_name,
            "folder": folder_name,
            "matches": 0,
            "total": len(image_paths),
            "rate": "错误",
            "output": str(e),
        })
    
    print()

# 打印汇总结果
print("\n" + "=" * 80)
print("测试完成 - 结果汇总")
print("=" * 80)
print(f"\n{'商品':<20} {'匹配率':<15} {'结果':<10}")
print("-" * 80)

for r in results_summary:
    status = "✓" if r["matches"] > 0 else "⊘"
    print(f"{status} {r['product']:<18} {r['matches']}/{r['total']} ({r['rate']:<6}) {r['output']}")

print("\n" + "=" * 80)
print(f"总计: {len([r for r in results_summary if r['matches'] > 0])}/{len(PRODUCTS)} 个商品成功处理")
print("=" * 80)
