"""
EAN / 360° Product Segmentation System

Core package: config, pipeline, segmentation (SAM2), embedding (DINOv2), matching, export.
"""

from . import config
from .pipeline import ProductSegmentationPipeline, PipelineResult
from .image_utils import load_image, load_images_from_directory

__all__ = [
    "config",
    "ProductSegmentationPipeline",
    "PipelineResult",
    "load_image",
    "load_images_from_directory",
]
