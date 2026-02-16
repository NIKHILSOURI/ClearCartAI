from __future__ import annotations
from typing import List
import numpy as np
from ultralytics import SAM

class SamSegmenter:
    """
    SAM 3 via Ultralytics SAM interface.
    Uses visual prompts: points + labels (1=positive, 0=negative).
    """
    def __init__(self, model_path: str = "sam3.pt"):
        self.model = SAM(model_path)

    def segment_with_points(self, image, points: List[List[int]], labels: List[int]) -> np.ndarray:
        results = self.model(image, points=points, labels=labels)
        r0 = results[0]
        if r0.masks is None:
            return np.zeros((0, 1, 1), dtype=np.uint8)
        return r0.masks.data.cpu().numpy()  # (N,H,W)
