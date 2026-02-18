"""
Cross-Image Instance Matcher

Given a reference embedding (from the user-selected product) and a target image,
finds the same product instance in the target image by:
1. Generating candidate proposals with SAM2
2. Computing FFA embeddings for each proposal
3. Ranking by cosine similarity to the reference
4. Applying NMS to remove duplicate detections
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
import config
from sam2_segmenter import SAM2AutoSegmenter, mask_to_bbox, bbox_iou
from dinov2_embedder import DINOv2Embedder


@dataclass
class MatchResult:
    """A single matched instance in a target image."""
    mask: np.ndarray           # (H, W) boolean mask
    bbox: np.ndarray           # [x1, y1, x2, y2]
    similarity: float          # Cosine similarity to reference
    predicted_iou: float       # SAM2's IoU prediction
    area: int                  # Mask area in pixels
    embedding: np.ndarray      # FFA embedding vector


@dataclass
class ImageMatchResults:
    """All matches found in a single target image."""
    image_path: str
    image: np.ndarray
    matches: List[MatchResult] = field(default_factory=list)

    @property
    def best_match(self) -> Optional[MatchResult]:
        """Return highest-similarity match, or None."""
        return self.matches[0] if self.matches else None

    @property
    def has_match(self) -> bool:
        return len(self.matches) > 0


class InstanceMatcher:
    """Match a reference product across multiple images."""

    def __init__(
        self,
        similarity_threshold: float = None,
        top_k: int = None,
        nms_iou_threshold: float = None,
    ):
        self.auto_segmenter = SAM2AutoSegmenter()
        self.embedder = DINOv2Embedder()

        self.similarity_threshold = similarity_threshold or config.SIMILARITY_THRESHOLD
        self.top_k = top_k or config.TOP_K_MATCHES
        self.nms_iou_threshold = nms_iou_threshold or config.NMS_IOU_THRESHOLD

    def match_in_image(
        self,
        target_image: np.ndarray,
        reference_embedding: np.ndarray,
        image_path: str = "",
    ) -> ImageMatchResults:
        """Find the reference product in a target image.

        Args:
            target_image: RGB numpy array (H, W, 3)
            reference_embedding: L2-normalized FFA embedding of the reference product
            image_path: Path string for bookkeeping

        Returns:
            ImageMatchResults with all matches above threshold
        """
        results = ImageMatchResults(image_path=image_path, image=target_image)

        # Step 1: Generate candidate proposals
        proposals = self.auto_segmenter.generate_proposals(target_image)

        if not proposals:
            print(f"[Matcher] No proposals generated for {image_path}")
            return results

        # Step 2: Compute FFA embeddings for all proposals (batched, single forward pass)
        proposal_masks = [p['segmentation'] for p in proposals]
        proposal_embeddings = self.embedder.compute_batch_ffa_embeddings(
            target_image, proposal_masks
        )

        # Step 3: Compute similarities
        similarities = self.embedder.batch_cosine_similarity(
            reference_embedding, proposal_embeddings
        )

        # Step 4: Filter by threshold and collect candidates
        candidates = []
        for i, (sim, proposal) in enumerate(zip(similarities, proposals)):
            if sim >= self.similarity_threshold:
                mask = proposal['segmentation']
                bbox = mask_to_bbox(mask)
                candidates.append(MatchResult(
                    mask=mask,
                    bbox=bbox,
                    similarity=float(sim),
                    predicted_iou=proposal['predicted_iou'],
                    area=proposal['area'],
                    embedding=proposal_embeddings[i],
                ))

        # Step 5: Sort by similarity (descending)
        candidates.sort(key=lambda x: x.similarity, reverse=True)

        # Step 6: Apply NMS (keep top-scoring, remove overlapping)
        kept = self._nms(candidates)

        # Step 7: Keep top-K
        results.matches = kept[:self.top_k]

        print(f"[Matcher] {image_path}: {len(proposals)} proposals → "
              f"{len(candidates)} above threshold → {len(results.matches)} after NMS")

        return results

    def _nms(self, candidates: List[MatchResult]) -> List[MatchResult]:
        """Non-maximum suppression based on bounding box IoU.

        Candidates should already be sorted by similarity (descending).
        """
        if not candidates:
            return []

        kept = []
        suppressed = set()

        for i, cand in enumerate(candidates):
            if i in suppressed:
                continue

            kept.append(cand)

            # Suppress lower-scoring candidates that overlap too much
            for j in range(i + 1, len(candidates)):
                if j in suppressed:
                    continue
                iou = bbox_iou(cand.bbox, candidates[j].bbox)
                if iou > self.nms_iou_threshold:
                    suppressed.add(j)

        return kept

    def match_across_images(
        self,
        target_images: List[Tuple[str, np.ndarray]],
        reference_embedding: np.ndarray,
    ) -> List[ImageMatchResults]:
        """Match reference product across multiple target images.

        Args:
            target_images: List of (path, image_array) tuples
            reference_embedding: FFA embedding of the reference product

        Returns:
            List of ImageMatchResults, one per target image
        """
        all_results = []

        for i, (path, image) in enumerate(target_images):
            print(f"\n[Matcher] Processing image {i+1}/{len(target_images)}: {path}")
            result = self.match_in_image(image, reference_embedding, path)
            all_results.append(result)

        # Summary
        matched = sum(1 for r in all_results if r.has_match)
        print(f"\n[Matcher] Summary: Found product in {matched}/{len(all_results)} images")

        return all_results
