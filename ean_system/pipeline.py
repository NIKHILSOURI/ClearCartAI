"""
End-to-End Product Segmentation Pipeline

Orchestrates the full flow:
1. Load 360° images
2. Interactive selection of reference product (or programmatic bbox)
3. SAM2 segmentation of reference
4. DINOv2 FFA embedding of reference
5. Cross-image matching on all target images
6. Export results (masks, cutouts, visualizations)
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Union
from dataclasses import dataclass

from . import config
from .sam2_segmenter import SAM2InteractiveSegmenter, mask_to_bbox
from .dinov2_embedder import DINOv2Embedder
from .matcher import InstanceMatcher, ImageMatchResults, MatchResult
from .image_utils import load_image, load_images_from_directory
from .export import ResultExporter


@dataclass
class PipelineResult:
    """Complete results from the pipeline."""
    reference_image_path: str
    reference_mask: np.ndarray
    reference_embedding: np.ndarray
    reference_bbox: np.ndarray
    image_results: List[ImageMatchResults]

    @property
    def total_matches(self) -> int:
        return sum(1 for r in self.image_results if r.has_match)

    @property
    def total_images(self) -> int:
        return len(self.image_results)


class ProductSegmentationPipeline:
    """Main pipeline class — the single entry point for users."""

    def __init__(
        self,
        similarity_threshold: float = None,
        top_k: int = None,
    ):
        """Initialize pipeline components.

        Models are loaded lazily on first use.
        """
        self.interactive_segmenter = None  # Lazy
        self.embedder = None               # Lazy
        self.matcher = None                # Lazy
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    def _ensure_models(self):
        """Lazy-load all models."""
        if self.interactive_segmenter is None:
            self.interactive_segmenter = SAM2InteractiveSegmenter()
        if self.embedder is None:
            self.embedder = DINOv2Embedder()
        if self.matcher is None:
            self.matcher = InstanceMatcher(
                similarity_threshold=self.similarity_threshold,
                top_k=self.top_k,
            )

    # ─── Primary Entry Point: Point Prompt ────────────────────

    def run_with_point(
        self,
        image_paths: List[str],
        reference_image_path: str,
        reference_point: Tuple[int, int],
        point_label: int = 1,
        output_dir: str = None,
    ) -> PipelineResult:
        """Run full pipeline with a point click on the reference image.

        Args:
            image_paths: All image paths (including reference)
            reference_image_path: Path to the reference image
            reference_point: (x, y) coordinate of the click on the product
            point_label: 1 for positive (click on product), 0 for negative
            output_dir: Where to save results (None = don't save)

        Returns:
            PipelineResult with all matches
        """
        self._ensure_models()
        config.print_config()

        # Load reference image
        print(f"\n{'='*60}")
        print(f"Phase 1: Reference Product Segmentation")
        print(f"{'='*60}")
        ref_image = load_image(reference_image_path)
        print(f"Reference image: {reference_image_path} ({ref_image.shape})")

        # SAM2 segmentation with point prompt
        self.interactive_segmenter.set_image(ref_image)
        point_coords = np.array([[reference_point[0], reference_point[1]]])
        point_labels = np.array([point_label])

        masks, scores, logits = self.interactive_segmenter.segment_with_points(
            point_coords, point_labels, multimask_output=True
        )

        # Pick the best mask (highest score)
        best_idx = np.argmax(scores)
        ref_mask = masks[best_idx]
        ref_bbox = mask_to_bbox(ref_mask)
        print(f"Selected mask {best_idx} with score {scores[best_idx]:.3f}")
        print(f"Mask area: {ref_mask.sum()} pixels, bbox: {ref_bbox}")

        # Compute reference FFA embedding
        print(f"\n{'='*60}")
        print(f"Phase 2: Computing Reference Embedding (DINOv2 FFA)")
        print(f"{'='*60}")
        ref_embedding = self.embedder.compute_ffa_embedding(ref_image, ref_mask)
        print(f"Reference embedding shape: {ref_embedding.shape}, norm: {np.linalg.norm(ref_embedding):.4f}")

        # Match across all target images
        print(f"\n{'='*60}")
        print(f"Phase 3: Cross-Image Matching")
        print(f"{'='*60}")
        target_images = []
        for path in image_paths:
            if Path(path).resolve() != Path(reference_image_path).resolve():
                img = load_image(path)
                target_images.append((path, img))

        print(f"Matching across {len(target_images)} target images...")
        image_results = self.matcher.match_across_images(
            target_images, ref_embedding
        )

        result = PipelineResult(
            reference_image_path=reference_image_path,
            reference_mask=ref_mask,
            reference_embedding=ref_embedding,
            reference_bbox=ref_bbox,
            image_results=image_results,
        )

        # Export
        if output_dir:
            print(f"\n{'='*60}")
            print(f"Phase 4: Exporting Results")
            print(f"{'='*60}")
            exporter = ResultExporter(output_dir)
            exporter.export_all(result, ref_image)

        # Summary
        print(f"\n{'='*60}")
        print(f"RESULTS SUMMARY")
        print(f"{'='*60}")
        print(f"Reference: {reference_image_path}")
        print(f"Product found in: {result.total_matches}/{result.total_images} target images")
        for r in image_results:
            status = "✓" if r.has_match else "✗"
            sim = f"{r.best_match.similarity:.3f}" if r.best_match else "N/A"
            print(f"  {status} {Path(r.image_path).name}: similarity={sim}")

        return result

    # ─── Alternative Entry: Bounding Box Prompt ───────────────

    def run_with_bbox(
        self,
        image_paths: List[str],
        reference_image_path: str,
        reference_bbox: Tuple[int, int, int, int],
        output_dir: str = None,
    ) -> PipelineResult:
        """Run with bounding box prompt instead of point click.

        Args:
            reference_bbox: (x1, y1, x2, y2) bounding box around the product
        """
        self._ensure_models()
        config.print_config()

        ref_image = load_image(reference_image_path)

        # SAM2 segmentation with box prompt
        self.interactive_segmenter.set_image(ref_image)
        box = np.array(reference_bbox)

        masks, scores, logits = self.interactive_segmenter.segment_with_box(
            box, multimask_output=False
        )

        ref_mask = masks[0]
        ref_bbox = mask_to_bbox(ref_mask)

        # FFA embedding
        ref_embedding = self.embedder.compute_ffa_embedding(ref_image, ref_mask)

        # Match
        target_images = []
        for path in image_paths:
            if Path(path).resolve() != Path(reference_image_path).resolve():
                img = load_image(path)
                target_images.append((path, img))

        image_results = self.matcher.match_across_images(
            target_images, ref_embedding
        )

        result = PipelineResult(
            reference_image_path=reference_image_path,
            reference_mask=ref_mask,
            reference_embedding=ref_embedding,
            reference_bbox=ref_bbox,
            image_results=image_results,
        )

        if output_dir:
            exporter = ResultExporter(output_dir)
            exporter.export_all(result, ref_image)

        return result

    # ─── Interactive Entry (with UI) ──────────────────────────

    def run_interactive(
        self,
        image_paths: List[str],
        reference_image_path: str = None,
        output_dir: str = None,
    ) -> PipelineResult:
        """Run with interactive click-to-select UI.

        Opens a matplotlib window where you click on the product.

        Args:
            image_paths: All image paths
            reference_image_path: Which image to use as reference (default: first)
            output_dir: Where to save results
        """
        from .interactive_selector import InteractiveSelector

        self._ensure_models()

        if reference_image_path is None:
            reference_image_path = image_paths[0]

        ref_image = load_image(reference_image_path)

        # Launch interactive selector
        selector = InteractiveSelector(self.interactive_segmenter)
        ref_mask, ref_bbox = selector.select_product(ref_image, reference_image_path)

        if ref_mask is None:
            print("No product selected. Exiting.")
            return None

        # Compute embedding and match
        ref_embedding = self.embedder.compute_ffa_embedding(ref_image, ref_mask)

        target_images = []
        for path in image_paths:
            if Path(path).resolve() != Path(reference_image_path).resolve():
                img = load_image(path)
                target_images.append((path, img))

        image_results = self.matcher.match_across_images(
            target_images, ref_embedding
        )

        result = PipelineResult(
            reference_image_path=reference_image_path,
            reference_mask=ref_mask,
            reference_embedding=ref_embedding,
            reference_bbox=ref_bbox,
            image_results=image_results,
        )

        if output_dir:
            exporter = ResultExporter(output_dir)
            exporter.export_all(result, ref_image)

        return result
