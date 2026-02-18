"""
Export pipeline results: masks, cutouts, visualizations, and COCO format.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional
from PIL import Image
from datetime import datetime

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import config
from image_utils import apply_mask_overlay, crop_with_mask, draw_bbox


class ResultExporter:
    """Export pipeline results to various formats."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.masks_dir = self.output_dir / "masks"
        self.cutouts_dir = self.output_dir / "cutouts"
        self.overlays_dir = self.output_dir / "overlays"
        self.masks_dir.mkdir(exist_ok=True)
        self.cutouts_dir.mkdir(exist_ok=True)
        self.overlays_dir.mkdir(exist_ok=True)

    def export_all(self, pipeline_result, reference_image: np.ndarray):
        """Export everything from a PipelineResult.

        Args:
            pipeline_result: PipelineResult from the pipeline
            reference_image: The reference image array
        """
        # Export reference
        self._export_reference(
            reference_image,
            pipeline_result.reference_mask,
            pipeline_result.reference_bbox,
            pipeline_result.reference_image_path,
        )

        # Export each target image result
        for result in pipeline_result.image_results:
            self._export_image_result(result)

        # Export gallery visualization
        self._export_gallery(pipeline_result, reference_image)

        # Export summary JSON
        self._export_summary(pipeline_result)

        print(f"[Export] All results saved to {self.output_dir}")

    def _export_reference(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        bbox: np.ndarray,
        path: str,
    ):
        """Export reference image results."""
        stem = Path(path).stem

        # Save mask as PNG
        mask_img = Image.fromarray((mask * 255).astype(np.uint8))
        mask_img.save(self.masks_dir / f"ref_{stem}_mask.png")

        # Save cutout (RGBA with transparent background)
        cutout = crop_with_mask(image, mask, transparent_bg=True)
        cutout_img = Image.fromarray(cutout)
        cutout_img.save(self.cutouts_dir / f"ref_{stem}_cutout.png")

        # Save overlay visualization
        overlay = apply_mask_overlay(image, mask, config.MASK_COLOR, config.MASK_ALPHA)
        overlay = draw_bbox(overlay, bbox, color=(0, 255, 0), label="Reference")
        overlay_img = Image.fromarray(overlay)
        overlay_img.save(self.overlays_dir / f"ref_{stem}_overlay.jpg")

        print(f"[Export] Reference exported: {stem}")

    def _export_image_result(self, result):
        """Export results for a single target image."""
        stem = Path(result.image_path).stem

        if not result.has_match:
            # Save unmatched overlay (just the original image)
            overlay_img = Image.fromarray(result.image)
            overlay_img.save(self.overlays_dir / f"{stem}_no_match.jpg")
            return

        for i, match in enumerate(result.matches):
            suffix = f"_match{i}" if len(result.matches) > 1 else ""

            # Mask
            mask_img = Image.fromarray((match.mask * 255).astype(np.uint8))
            mask_img.save(self.masks_dir / f"{stem}{suffix}_mask.png")

            # Cutout
            cutout = crop_with_mask(result.image, match.mask, transparent_bg=True)
            cutout_img = Image.fromarray(cutout)
            cutout_img.save(self.cutouts_dir / f"{stem}{suffix}_cutout.png")

            # Overlay
            overlay = apply_mask_overlay(
                result.image, match.mask, config.MASK_COLOR, config.MASK_ALPHA
            )
            label = f"sim={match.similarity:.2f}"
            overlay = draw_bbox(overlay, match.bbox, color=(0, 255, 0), label=label)
            overlay_img = Image.fromarray(overlay)
            overlay_img.save(self.overlays_dir / f"{stem}{suffix}_overlay.jpg")

        print(f"[Export] {stem}: {len(result.matches)} match(es) exported")

    def _export_gallery(self, pipeline_result, reference_image: np.ndarray):
        """Create a gallery visualization of all results."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            import matplotlib.pyplot as plt

            n_images = 1 + len(pipeline_result.image_results)
            cols = min(config.GALLERY_COLS, n_images)
            rows = (n_images + cols - 1) // cols

            fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
            if rows == 1 and cols == 1:
                axes = np.array([[axes]])
            elif rows == 1:
                axes = axes[np.newaxis, :]
            elif cols == 1:
                axes = axes[:, np.newaxis]

            # Reference image
            ax = axes[0, 0]
            overlay = apply_mask_overlay(
                reference_image, pipeline_result.reference_mask,
                config.MASK_COLOR, config.MASK_ALPHA
            )
            ax.imshow(overlay)
            ax.set_title("REFERENCE", fontsize=10, fontweight='bold', color='green')
            ax.axis('off')

            # Target images
            for idx, result in enumerate(pipeline_result.image_results):
                r, c = divmod(idx + 1, cols)
                ax = axes[r, c]

                if result.has_match:
                    match = result.best_match
                    overlay = apply_mask_overlay(
                        result.image, match.mask,
                        config.MASK_COLOR, config.MASK_ALPHA
                    )
                    ax.imshow(overlay)
                    ax.set_title(
                        f"{Path(result.image_path).name}\nsim={match.similarity:.3f}",
                        fontsize=8, color='green'
                    )
                else:
                    ax.imshow(result.image)
                    ax.set_title(
                        f"{Path(result.image_path).name}\nNo match",
                        fontsize=8, color='red'
                    )
                ax.axis('off')

            # Hide empty axes
            for idx in range(n_images, rows * cols):
                r, c = divmod(idx, cols)
                axes[r, c].axis('off')

            plt.tight_layout()
            plt.savefig(
                self.output_dir / "gallery.jpg",
                dpi=150, bbox_inches='tight', pad_inches=0.2
            )
            plt.close()
            print("[Export] Gallery saved: gallery.jpg")

        except Exception as e:
            print(f"[Export] Warning: Could not create gallery: {e}")

    def _export_summary(self, pipeline_result):
        """Export a JSON summary of all results."""
        summary = {
            "timestamp": datetime.now().isoformat(),
            "reference": {
                "image": pipeline_result.reference_image_path,
                "mask_area": int(pipeline_result.reference_mask.sum()),
                "bbox": pipeline_result.reference_bbox.tolist(),
            },
            "config": {
                "similarity_threshold": config.SIMILARITY_THRESHOLD,
                "sam2_checkpoint": config.SAM2_CHECKPOINT,
                "dinov2_model": config.DINOV2_MODEL_NAME,
            },
            "results": [],
        }

        for result in pipeline_result.image_results:
            entry = {
                "image": result.image_path,
                "matched": result.has_match,
                "matches": [],
            }
            for match in result.matches:
                entry["matches"].append({
                    "similarity": round(match.similarity, 4),
                    "predicted_iou": round(match.predicted_iou, 4),
                    "area": match.area,
                    "bbox": match.bbox.tolist(),
                })
            summary["results"].append(entry)

        summary["total_images"] = pipeline_result.total_images
        summary["total_matches"] = pipeline_result.total_matches
        summary["match_rate"] = round(
            pipeline_result.total_matches / max(1, pipeline_result.total_images), 3
        )

        with open(self.output_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("[Export] Summary saved: summary.json")
