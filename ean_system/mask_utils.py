"""
Mask processing utilities: refinement, morphological operations, format conversion.
"""

import numpy as np
from typing import List, Tuple, Optional
import cv2


def refine_mask_morphological(
    mask: np.ndarray,
    close_kernel: int = 5,
    open_kernel: int = 3,
    iterations: int = 2,
) -> np.ndarray:
    """Apply morphological closing + opening to clean up a mask.

    Closing fills small holes, opening removes small noise.

    Args:
        mask: Boolean mask (H, W)
        close_kernel: Kernel size for closing operation
        open_kernel: Kernel size for opening operation
        iterations: Number of iterations

    Returns:
        Refined boolean mask
    """
    mask_uint8 = mask.astype(np.uint8) * 255

    # Closing (fill holes)
    kernel_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (close_kernel, close_kernel)
    )
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel_close, iterations=iterations)

    # Opening (remove noise)
    kernel_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (open_kernel, open_kernel)
    )
    mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel_open, iterations=iterations)

    return mask_uint8 > 127


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component in a mask.

    Useful when SAM2 produces a mask with disconnected regions.

    Args:
        mask: Boolean mask (H, W)

    Returns:
        Boolean mask with only the largest component
    """
    mask_uint8 = mask.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask_uint8, connectivity=8
    )

    if num_labels <= 1:
        return mask

    # Label 0 is background, find largest non-background component
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = np.argmax(areas) + 1

    return labels == largest_label


def mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection over Union between two boolean masks."""
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def mask_to_polygon(mask: np.ndarray) -> List[np.ndarray]:
    """Convert a boolean mask to polygon contours.

    Returns:
        List of contour arrays, each shape (N, 1, 2)
    """
    mask_uint8 = mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def mask_to_rle(mask: np.ndarray) -> dict:
    """Convert a boolean mask to Run-Length Encoding (COCO format).

    Returns:
        dict with 'counts' and 'size' keys
    """
    pixels = mask.T.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = np.concatenate([[0], runs, [len(pixels)]])
    lengths = runs[1:] - runs[:-1]

    if pixels[0]:
        # Starts with 1 — prepend 0-length run for COCO format
        lengths = np.concatenate([[0], lengths])

    return {
        'counts': lengths.tolist(),
        'size': [mask.shape[0], mask.shape[1]],
    }


def smooth_mask_boundary(mask: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """Smooth mask boundaries using Gaussian blur + re-threshold.

    Produces cleaner edges for export/visualization.
    """
    mask_float = mask.astype(np.float32)
    blurred = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
    return blurred > 0.5


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """Combine multiple boolean masks into one (logical OR)."""
    if not masks:
        raise ValueError("No masks to combine")
    result = masks[0].copy()
    for m in masks[1:]:
        result = np.logical_or(result, m)
    return result
