"""
Image loading and processing utilities.
Handles 360° camera images, various formats, and preprocessing.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
from PIL import Image
import glob


SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}


def load_image(path: str, max_size: int = None) -> np.ndarray:
    """Load an image as RGB numpy array.

    Args:
        path: Path to image file
        max_size: If set, resize longest edge to this value

    Returns:
        RGB numpy array (H, W, 3), dtype uint8
    """
    img = Image.open(path).convert("RGB")

    if max_size and max(img.size) > max_size:
        ratio = max_size / max(img.size)
        new_size = (int(img.width * ratio), int(img.height * ratio))
        img = img.resize(new_size, Image.LANCZOS)

    return np.array(img)


def load_images_from_directory(
    directory: str,
    pattern: str = "*",
    max_size: int = None,
    sort: bool = True,
) -> List[Tuple[str, np.ndarray]]:
    """Load all images from a directory.

    Args:
        directory: Path to directory
        pattern: Glob pattern (e.g., "*.jpg")
        max_size: Max image dimension
        sort: Sort paths alphabetically

    Returns:
        List of (path, image_array) tuples
    """
    dir_path = Path(directory)
    paths = []

    for ext in SUPPORTED_EXTENSIONS:
        paths.extend(dir_path.glob(f"{pattern}{ext}"))
        paths.extend(dir_path.glob(f"{pattern}{ext.upper()}"))

    if sort:
        paths = sorted(set(paths))

    images = []
    for p in paths:
        try:
            img = load_image(str(p), max_size=max_size)
            images.append((str(p), img))
        except Exception as e:
            print(f"[Warning] Failed to load {p}: {e}")

    print(f"Loaded {len(images)} images from {directory}")
    return images


def load_images_from_paths(
    paths: List[str],
    max_size: int = None,
) -> List[Tuple[str, np.ndarray]]:
    """Load images from a list of paths.

    Args:
        paths: List of image file paths
        max_size: Max image dimension

    Returns:
        List of (path, image_array) tuples
    """
    images = []
    for p in paths:
        try:
            img = load_image(p, max_size=max_size)
            images.append((p, img))
        except Exception as e:
            print(f"[Warning] Failed to load {p}: {e}")
    return images


def apply_mask_overlay(
    image: np.ndarray,
    mask: np.ndarray,
    color: Tuple[float, float, float] = (0.2, 0.8, 0.2),
    alpha: float = 0.45,
) -> np.ndarray:
    """Apply a colored semi-transparent mask overlay on an image.

    Args:
        image: RGB numpy array (H, W, 3), uint8
        mask: Boolean mask (H, W)
        color: RGB color tuple (0-1 range)
        alpha: Overlay transparency

    Returns:
        Image with mask overlay, uint8
    """
    overlay = image.copy().astype(np.float32)
    mask_color = np.array(color) * 255
    
    # Ensure mask is boolean type
    mask_bool = mask.astype(bool)

    overlay[mask_bool] = overlay[mask_bool] * (1 - alpha) + mask_color * alpha
    return overlay.clip(0, 255).astype(np.uint8)


def crop_with_mask(
    image: np.ndarray,
    mask: np.ndarray,
    padding: int = 10,
    transparent_bg: bool = True,
) -> np.ndarray:
    """Crop the masked region from an image.

    Args:
        image: RGB numpy array (H, W, 3)
        mask: Boolean mask (H, W)
        padding: Pixels of padding around the bounding box
        transparent_bg: If True, return RGBA with transparent background

    Returns:
        Cropped image (RGBA if transparent_bg, else RGB)
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        return image

    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]

    # Add padding
    h, w = image.shape[:2]
    y1 = max(0, y1 - padding)
    y2 = min(h - 1, y2 + padding)
    x1 = max(0, x1 - padding)
    x2 = min(w - 1, x2 + padding)

    cropped = image[y1:y2+1, x1:x2+1].copy()

    if transparent_bg:
        # Create RGBA with transparent background outside mask
        mask_crop = mask[y1:y2+1, x1:x2+1]
        rgba = np.zeros((*cropped.shape[:2], 4), dtype=np.uint8)
        rgba[..., :3] = cropped
        rgba[..., 3] = mask_crop.astype(np.uint8) * 255
        return rgba

    return cropped


def draw_bbox(
    image: np.ndarray,
    bbox: np.ndarray,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    label: str = None,
) -> np.ndarray:
    """Draw a bounding box on an image.

    Args:
        image: RGB numpy array
        bbox: [x1, y1, x2, y2]
        color: BGR color tuple
        thickness: Line thickness
        label: Optional text label

    Returns:
        Image with bounding box drawn
    """
    import cv2

    img = image.copy()
    x1, y1, x2, y2 = bbox.astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, 1)
        cv2.rectangle(img, (x1, y1 - text_h - 8), (x1 + text_w + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), font, font_scale, (255, 255, 255), 1)

    return img
