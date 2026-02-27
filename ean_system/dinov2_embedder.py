"""
DINOv2 Feature Extraction with Foreground Feature Averaging (FFA)

This is the key module that enables cross-image instance matching.
Instead of using the global CLS token, we:
1. Extract ALL patch embeddings from DINOv2
2. Use SAM2's mask to identify which patches belong to the foreground object
3. Average only those foreground patch embeddings
4. L2-normalize to produce a unit-vector instance embedding

This produces embeddings where:
- Same product from different 360° angles → high cosine similarity (>0.8)
- Different products → low cosine similarity (<0.5)
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple
from PIL import Image

from . import config
from .model_loader import ModelLoader


class DINOv2Embedder:
    """Extract instance embeddings using DINOv2 + mask-guided FFA."""

    def __init__(self):
        self.model, self.processor = ModelLoader.get_dinov2()
        self.device = config.get_device()
        self.patch_size = config.DINOV2_PATCH_SIZE
        self.embedding_dim = config.DINOV2_EMBEDDING_DIM

    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for DINOv2.

        Args:
            image: RGB numpy array (H, W, 3), uint8

        Returns:
            Preprocessed tensor ready for the model
        """
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        return inputs["pixel_values"].to(self.device)

    def _get_patch_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Extract all patch embeddings (excluding CLS token).

        Args:
            pixel_values: (1, 3, H, W) preprocessed image tensor

        Returns:
            patch_embeddings: (1, num_patches, embedding_dim)
        """
        with torch.inference_mode():
            outputs = self.model(pixel_values, output_hidden_states=True)
            # last_hidden_state shape: (batch, 1 + num_patches, dim)
            # Index 0 is CLS token (or register tokens), rest are patches
            last_hidden = outputs.last_hidden_state

            # DINOv2 with registers: first token is CLS, then register tokens,
            # then patch tokens. For models with registers, we need to handle
            # the offset. The processor handles input size → we can compute
            # expected number of patches from the hidden state.
            #
            # For dinov2-vitl14-reg: CLS + 4 registers + patches
            # For dinov2-vitl14: CLS + patches
            #
            # Safe approach: compute expected grid size from input dims
            h, w = pixel_values.shape[2], pixel_values.shape[3]
            num_patches_h = h // self.patch_size
            num_patches_w = w // self.patch_size
            expected_patches = num_patches_h * num_patches_w

            # Take the LAST `expected_patches` tokens (patches are always at the end)
            patch_embeddings = last_hidden[:, -expected_patches:, :]

        return patch_embeddings, num_patches_h, num_patches_w

    def compute_ffa_embedding(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        return_patch_info: bool = False,
    ) -> np.ndarray:
        """Compute Foreground Feature Averaging embedding for a masked object.

        This is the core function. Given an image and a binary mask indicating
        the object of interest, it:
        1. Runs the image through DINOv2
        2. Maps the mask to the patch grid
        3. Averages patch embeddings where the mask overlaps
        4. L2-normalizes the result

        Args:
            image: RGB numpy array (H, W, 3), uint8
            mask: Boolean mask (H, W), True = foreground
            return_patch_info: If True, also return patch-level details

        Returns:
            embedding: L2-normalized embedding vector (embedding_dim,)
            (optional) patch_mask: which patches were included
        """
        # Preprocess and extract patches
        pixel_values = self._preprocess_image(image)
        patch_embeddings, grid_h, grid_w = self._get_patch_embeddings(pixel_values)

        # Reshape patch embeddings to spatial grid
        # patch_embeddings: (1, grid_h * grid_w, dim) → (1, grid_h, grid_w, dim)
        patches_spatial = patch_embeddings.view(1, grid_h, grid_w, -1)

        # Resize mask to match patch grid
        # The mask is at original image resolution; we need it at patch resolution
        mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        mask_resized = F.interpolate(
            mask_tensor,
            size=(grid_h, grid_w),
            mode='bilinear',
            align_corners=False,
        ).squeeze()  # (grid_h, grid_w)

        # Binarize: patches with >50% mask coverage count as foreground
        patch_mask = (mask_resized > 0.5).to(self.device)

        # Safety: if no patches are in the foreground, fall back to full image
        if patch_mask.sum() == 0:
            print("[DINOv2] Warning: No foreground patches found, using full image")
            patch_mask = torch.ones(grid_h, grid_w, dtype=torch.bool, device=self.device)

        # Extract and average foreground patch embeddings
        fg_patches = patches_spatial[0][patch_mask]  # (num_fg_patches, dim)
        embedding = fg_patches.mean(dim=0)  # (dim,)

        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=0)

        result = embedding.cpu().numpy()

        if return_patch_info:
            return result, patch_mask.cpu().numpy()
        return result

    def compute_batch_ffa_embeddings(
        self,
        image: np.ndarray,
        masks: List[np.ndarray],
    ) -> np.ndarray:
        """Compute FFA embeddings for multiple masks on the SAME image.

        Much more efficient than calling compute_ffa_embedding() per mask,
        because the DINOv2 forward pass only runs once.

        Args:
            image: RGB numpy array (H, W, 3), uint8
            masks: List of boolean masks (H, W), one per proposal

        Returns:
            embeddings: (N, embedding_dim) array, one row per mask
        """
        if not masks:
            return np.zeros((0, self.embedding_dim))

        # Single forward pass for the image
        pixel_values = self._preprocess_image(image)
        patch_embeddings, grid_h, grid_w = self._get_patch_embeddings(pixel_values)
        patches_spatial = patch_embeddings.view(1, grid_h, grid_w, -1)

        embeddings = []

        for mask in masks:
            # Resize mask to patch grid
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)
            mask_resized = F.interpolate(
                mask_tensor,
                size=(grid_h, grid_w),
                mode='bilinear',
                align_corners=False,
            ).squeeze()

            patch_mask = (mask_resized > 0.5).to(self.device)

            if patch_mask.sum() == 0:
                # Fall back to global average
                patch_mask = torch.ones(grid_h, grid_w, dtype=torch.bool, device=self.device)

            fg_patches = patches_spatial[0][patch_mask]
            emb = fg_patches.mean(dim=0)
            emb = F.normalize(emb, p=2, dim=0)
            embeddings.append(emb.cpu().numpy())

        return np.stack(embeddings, axis=0)

    @staticmethod
    def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between two L2-normalized embeddings.

        Since embeddings are already L2-normalized, cosine similarity
        is simply the dot product.

        Args:
            emb1: (dim,) first embedding
            emb2: (dim,) second embedding

        Returns:
            Cosine similarity in [-1, 1]
        """
        return float(np.dot(emb1, emb2))

    @staticmethod
    def batch_cosine_similarity(
        reference: np.ndarray,
        candidates: np.ndarray,
    ) -> np.ndarray:
        """Compute cosine similarity between a reference and multiple candidates.

        Args:
            reference: (dim,) reference embedding
            candidates: (N, dim) candidate embeddings

        Returns:
            similarities: (N,) cosine similarities
        """
        # Both are L2-normalized, so dot product = cosine similarity
        return candidates @ reference
