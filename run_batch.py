#!/usr/bin/env python3
"""
Batch Product Segmentation (Non-Interactive)

Provide a reference image + point or bounding box to segment the product
and match it across all other images without any UI.

Usage:
    # With point prompt:
    python run_batch.py \
        --image-dir ./360_captures/ \
        --reference img_001.jpg \
        --point 250 300 \
        --output ./results/

    # With bounding box prompt:
    python run_batch.py \
        --image-dir ./360_captures/ \
        --reference img_001.jpg \
        --bbox 120 80 340 450 \
        --output ./results/

    # With custom similarity threshold:
    python run_batch.py \
        --images *.jpg \
        --reference img_001.jpg \
        --point 250 300 \
        --threshold 0.7 \
        --output ./results/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from pipeline import ProductSegmentationPipeline
from image_utils import load_images_from_directory


def main():
    parser = argparse.ArgumentParser(
        description="Batch 360° Product Segmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image-dir", type=str, help="Directory of images")
    input_group.add_argument("--images", nargs="+", type=str, help="Image file paths")

    # Reference
    parser.add_argument("--reference", type=str, required=True, help="Reference image path")

    # Prompt (one required)
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument(
        "--point", nargs=2, type=int, metavar=("X", "Y"),
        help="Point prompt: x y coordinates on the product"
    )
    prompt_group.add_argument(
        "--bbox", nargs=4, type=int, metavar=("X1", "Y1", "X2", "Y2"),
        help="Bounding box prompt: x1 y1 x2 y2 around the product"
    )

    # Options
    parser.add_argument("--output", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--threshold", type=float, default=None, help="Similarity threshold")
    parser.add_argument("--top-k", type=int, default=None, help="Max matches per image")
    parser.add_argument("--max-size", type=int, default=None, help="Max image dimension")

    args = parser.parse_args()

    # Collect paths
    if args.image_dir:
        images = load_images_from_directory(args.image_dir, max_size=args.max_size)
        image_paths = [p for p, _ in images]
    else:
        image_paths = args.images

    if not image_paths:
        print("Error: No images found!")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")

    # Initialize pipeline
    pipeline = ProductSegmentationPipeline(
        similarity_threshold=args.threshold,
        top_k=args.top_k,
    )

    # Run with appropriate prompt
    if args.point:
        result = pipeline.run_with_point(
            image_paths=image_paths,
            reference_image_path=args.reference,
            reference_point=tuple(args.point),
            output_dir=args.output,
        )
    else:
        result = pipeline.run_with_bbox(
            image_paths=image_paths,
            reference_image_path=args.reference,
            reference_bbox=tuple(args.bbox),
            output_dir=args.output,
        )

    print(f"\nDone! Found product in {result.total_matches}/{result.total_images} images")
    print(f"Results saved to: {args.output}")


if __name__ == "__main__":
    main()
