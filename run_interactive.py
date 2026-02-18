#!/usr/bin/env python3
"""
Interactive Product Segmentation from 360° Camera Images

Usage:
    # From directory of images:
    python run_interactive.py --image-dir ./360_captures/ --output ./results/

    # From specific files:
    python run_interactive.py --images img1.jpg img2.jpg img3.jpg --output ./results/

    # With specific reference image:
    python run_interactive.py --image-dir ./captures/ --reference img_001.jpg --output ./results/

    # With custom threshold:
    python run_interactive.py --image-dir ./captures/ --threshold 0.7 --output ./results/
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline import ProductSegmentationPipeline
from image_utils import load_images_from_directory


def main():
    parser = argparse.ArgumentParser(
        description="Interactive 360° Product Segmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input sources (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--image-dir", type=str,
        help="Directory containing 360° capture images"
    )
    input_group.add_argument(
        "--images", nargs="+", type=str,
        help="List of image file paths"
    )

    # Options
    parser.add_argument(
        "--reference", type=str, default=None,
        help="Reference image path (default: first image)"
    )
    parser.add_argument(
        "--output", type=str, default="./outputs",
        help="Output directory (default: ./outputs)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Similarity threshold (default: 0.65)"
    )
    parser.add_argument(
        "--top-k", type=int, default=None,
        help="Max matches per image (default: 3)"
    )
    parser.add_argument(
        "--max-size", type=int, default=None,
        help="Resize images to max dimension (for memory savings)"
    )

    args = parser.parse_args()

    # Collect image paths
    if args.image_dir:
        images = load_images_from_directory(args.image_dir, max_size=args.max_size)
        image_paths = [p for p, _ in images]
    else:
        image_paths = args.images

    if not image_paths:
        print("Error: No images found!")
        sys.exit(1)

    print(f"Found {len(image_paths)} images")

    # Reference image
    reference = args.reference or image_paths[0]

    # Run pipeline
    pipeline = ProductSegmentationPipeline(
        similarity_threshold=args.threshold,
        top_k=args.top_k,
    )

    result = pipeline.run_interactive(
        image_paths=image_paths,
        reference_image_path=reference,
        output_dir=args.output,
    )

    if result:
        print(f"\nDone! Found product in {result.total_matches}/{result.total_images} images")
        print(f"Results saved to: {args.output}")
    else:
        print("Pipeline cancelled or no product selected.")


if __name__ == "__main__":
    main()
