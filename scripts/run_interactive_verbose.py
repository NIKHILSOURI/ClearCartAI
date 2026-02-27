#!/usr/bin/env python3
"""
Interactive product segmentation with verbose step-by-step output.

Usage:
    python scripts/run_interactive_verbose.py --image-dir ./my_images/ --output ./results/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

def main():
    parser = argparse.ArgumentParser(description="Interactive segmentation with verbose output")
    parser.add_argument("--image-dir", type=str, required=True, help="Directory containing images")
    parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.65, help="Similarity threshold")
    parser.add_argument("--top-k", type=int, default=3, help="Max matches per image")
    args = parser.parse_args()

    print("=" * 70)
    print("360° Product Segmentation - Interactive (verbose)")
    print("=" * 70)

    print("\n[1/5] Loading config...")
    from ean_system import config
    config.print_config()

    print("\n[2/5] Loading images...")
    from ean_system.image_utils import load_images_from_directory

    images = load_images_from_directory(args.image_dir)
    image_paths = [p for p, _ in images]

    if not image_paths:
        print("Error: No images found!")
        sys.exit(1)

    print(f"  Loaded {len(image_paths)} images")
    for i, path in enumerate(image_paths[:3], 1):
        print(f"    {i}. {Path(path).name}")
    if len(image_paths) > 3:
        print(f"    ... and {len(image_paths) - 3} more")

    print("\n[3/5] Initializing pipeline (first run may download ~10GB models)...")
    from ean_system.pipeline import ProductSegmentationPipeline

    pipeline = ProductSegmentationPipeline(
        similarity_threshold=args.threshold,
        top_k=args.top_k,
    )

    print("\n[4/5] Starting interactive UI...")
    print("=" * 70)
    print("Controls: Left-click = product point, Right-click = background, Enter = confirm, R = reset, Q = quit")
    print("=" * 70)

    result = pipeline.run_interactive(
        image_paths=image_paths,
        reference_image_path=image_paths[0],
        output_dir=args.output,
    )

    if result:
        print("\n" + "=" * 70)
        print("[5/5] Done!")
        print("=" * 70)
        print(f"  Matches: {result.total_matches}/{result.total_images} images")
        print(f"  Output: {args.output}")
    else:
        print("\nCancelled or no product selected.")


if __name__ == "__main__":
    main()
