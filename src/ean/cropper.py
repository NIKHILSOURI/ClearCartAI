from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
import cv2

@dataclass
class CropResult:
    mask_u8: np.ndarray
    crop_rgb: np.ndarray
    crop_rgba: np.ndarray

def _bbox_from_mask(mask01: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask01 > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

def choose_best_mask(masks: np.ndarray, pos_points: list[list[int]]) -> Optional[np.ndarray]:
    if masks is None or len(masks) == 0:
        return None
    areas = masks.reshape(masks.shape[0], -1).sum(axis=1)
    if pos_points:
        x, y = pos_points[0]
        contain = []
        for i in range(masks.shape[0]):
            m = masks[i]
            if 0 <= y < m.shape[0] and 0 <= x < m.shape[1] and m[y, x] > 0.5:
                contain.append(i)
        if contain:
            best = max(contain, key=lambda i: areas[i])
            return masks[best]
    return masks[int(np.argmax(areas))]

def make_crops(image_bgr: np.ndarray, mask_float: np.ndarray, pad_ratio: float = 0.08, max_side: int = 1024) -> CropResult:
    H, W = mask_float.shape[:2]
    mask01 = (mask_float > 0.5).astype(np.uint8)
    mask_u8 = (mask01 * 255).astype(np.uint8)

    bbox = _bbox_from_mask(mask01) or (0, 0, W-1, H-1)
    x1, y1, x2, y2 = bbox

    bw = x2 - x1 + 1
    bh = y2 - y1 + 1
    pad = int(max(bw, bh) * pad_ratio)
    x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
    x2 = min(W-1, x2 + pad); y2 = min(H-1, y2 + pad)

    crop_bgr = image_bgr[y1:y2+1, x1:x2+1].copy()
    crop_mask = mask01[y1:y2+1, x1:x2+1].copy()

    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    alpha = (crop_mask * 255).astype(np.uint8)
    crop_rgba = np.dstack([crop_rgb, alpha])

    white = np.full_like(crop_rgb, 255)
    crop_rgb_white = np.where(crop_mask[..., None] == 1, crop_rgb, white).astype(np.uint8)

    h, w = crop_rgb_white.shape[:2]
    scale = min(1.0, float(max_side) / float(max(h, w)))
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        crop_rgb_white = cv2.resize(crop_rgb_white, (nw, nh), interpolation=cv2.INTER_AREA)
        crop_rgba = cv2.resize(crop_rgba, (nw, nh), interpolation=cv2.INTER_AREA)

    return CropResult(mask_u8=mask_u8, crop_rgb=crop_rgb_white, crop_rgba=crop_rgba)
