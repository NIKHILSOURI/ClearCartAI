"""
Model loader with lazy initialization and caching.
Downloads checkpoints on first use, keeps models in GPU memory.
"""

import warnings
import torch
from pathlib import Path
from typing import Optional

from . import config

# SAM2 may warn when optional C++ extension (_C) is missing; safe to ignore (see INSTALL.md)
warnings.filterwarnings(
    "ignore",
    message=".*cannot import name '_C'.*",
    category=UserWarning,
    module="sam2",
)


class ModelLoader:
    """Singleton-style loader that caches models after first load."""

    _sam2_predictor = None
    _sam2_auto_generator = None
    _dinov2_model = None
    _dinov2_processor = None

    @classmethod
    def get_sam2_predictor(cls):
        """Load SAM2 image predictor (for interactive segmentation)."""
        if cls._sam2_predictor is None:
            print("[ModelLoader] Loading SAM2 Image Predictor...")
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor

            device = config.get_device()
            dtype = config.get_dtype()

            # Try loading from HuggingFace first (easier), fall back to local
            try:
                cls._sam2_predictor = SAM2ImagePredictor.from_pretrained(
                    config.SAM2_HF_MODEL_ID,
                    device=device,
                )
            except Exception:
                # Fall back to local checkpoint
                checkpoint_path = config.CHECKPOINT_DIR / config.SAM2_CHECKPOINT
                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"SAM2 checkpoint not found at {checkpoint_path}. "
                        f"Run: python -c \"from sam2.build_sam import build_sam2; "
                        f"build_sam2('{config.SAM2_CONFIG}', '{checkpoint_path}')\" "
                        f"or download from https://github.com/facebookresearch/sam2"
                    )
                sam2_model = build_sam2(
                    config.SAM2_CONFIG,
                    str(checkpoint_path),
                    device=str(device),
                )
                cls._sam2_predictor = SAM2ImagePredictor(sam2_model)

            print(f"[ModelLoader] SAM2 Predictor loaded on {device}")

        return cls._sam2_predictor

    @classmethod
    def get_sam2_auto_generator(cls):
        """Load SAM2 automatic mask generator (for proposal generation)."""
        if cls._sam2_auto_generator is None:
            print("[ModelLoader] Loading SAM2 Automatic Mask Generator...")
            from sam2.build_sam import build_sam2
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

            device = config.get_device()

            try:
                from sam2.sam2_image_predictor import SAM2ImagePredictor
                # Build from HuggingFace
                predictor = SAM2ImagePredictor.from_pretrained(
                    config.SAM2_HF_MODEL_ID,
                    device=device,
                )
                sam2_model = predictor.model
            except Exception:
                checkpoint_path = config.CHECKPOINT_DIR / config.SAM2_CHECKPOINT
                sam2_model = build_sam2(
                    config.SAM2_CONFIG,
                    str(checkpoint_path),
                    device=str(device),
                )

            cls._sam2_auto_generator = SAM2AutomaticMaskGenerator(
                model=sam2_model,
                points_per_side=config.SAM2_POINTS_PER_SIDE,
                pred_iou_thresh=config.SAM2_PRED_IOU_THRESH,
                stability_score_thresh=config.SAM2_STABILITY_SCORE_THRESH,
                crop_n_layers=config.SAM2_CROP_N_LAYERS,
                min_mask_region_area=config.SAM2_MIN_MASK_REGION_AREA,
            )

            print(f"[ModelLoader] SAM2 Auto Generator loaded on {device}")

        return cls._sam2_auto_generator

    @classmethod
    def get_dinov2(cls):
        """Load DINOv2 model and processor.

        Returns:
            tuple: (model, processor)
        """
        if cls._dinov2_model is None:
            print("[ModelLoader] Loading DINOv2...")
            from transformers import AutoImageProcessor, AutoModel

            device = config.get_device()

            cls._dinov2_processor = AutoImageProcessor.from_pretrained(
                config.DINOV2_MODEL_NAME
            )
            cls._dinov2_model = AutoModel.from_pretrained(
                config.DINOV2_MODEL_NAME
            ).to(device).eval()

            # Freeze all parameters
            for param in cls._dinov2_model.parameters():
                param.requires_grad = False

            print(f"[ModelLoader] DINOv2 ({config.DINOV2_MODEL_NAME}) loaded on {device}")

        return cls._dinov2_model, cls._dinov2_processor

    @classmethod
    def unload_all(cls):
        """Release all models from GPU memory."""
        cls._sam2_predictor = None
        cls._sam2_auto_generator = None
        cls._dinov2_model = None
        cls._dinov2_processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[ModelLoader] All models unloaded")
