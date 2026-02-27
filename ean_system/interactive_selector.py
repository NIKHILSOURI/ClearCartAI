"""
Interactive Product Selector

Opens a matplotlib window where the user clicks on the product to segment.
Supports:
- Left click: positive point (this is the product)
- Right click: negative point (this is NOT the product)
- Middle click / Enter: confirm selection
- 'r': reset all points
- 'q': quit without selecting
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from typing import Tuple, Optional

from . import config
from .image_utils import apply_mask_overlay


class InteractiveSelector:
    """Interactive click-to-segment UI using matplotlib."""

    def __init__(self, segmenter):
        """
        Args:
            segmenter: SAM2InteractiveSegmenter instance
        """
        self.segmenter = segmenter
        self._points = []
        self._labels = []
        self._current_mask = None
        self._current_logits = None
        self._confirmed = False
        self._cancelled = False

    def select_product(
        self,
        image: np.ndarray,
        image_path: str = "",
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Launch interactive selection UI.

        Args:
            image: RGB numpy array (H, W, 3)
            image_path: For display title

        Returns:
            (mask, bbox) or (None, None) if cancelled
        """
        self.segmenter.set_image(image)
        self._image = image
        self._points = []
        self._labels = []
        self._current_mask = None
        self._current_logits = None
        self._confirmed = False
        self._cancelled = False

        # Create figure
        fig, self._ax = plt.subplots(1, 1, figsize=config.INTERACTIVE_FIGSIZE)
        self._ax.imshow(image)
        self._ax.set_title(
            f"Click on product to segment | L-click: +point | R-click: -point | "
            f"Enter: confirm | R: reset | Q: quit\n{image_path}",
            fontsize=9,
        )
        self._ax.axis('off')

        # Connect events
        fig.canvas.mpl_connect('button_press_event', self._on_click)
        fig.canvas.mpl_connect('key_press_event', self._on_key)

        plt.tight_layout()
        plt.show(block=True)

        if self._confirmed and self._current_mask is not None:
            from .sam2_segmenter import mask_to_bbox
            bbox = mask_to_bbox(self._current_mask)
            return self._current_mask, bbox

        return None, None

    def _on_click(self, event):
        """Handle mouse clicks."""
        if event.inaxes != self._ax:
            return

        x, y = int(event.xdata), int(event.ydata)

        if event.button == 1:  # Left click = positive
            self._points.append([x, y])
            self._labels.append(1)
        elif event.button == 3:  # Right click = negative
            self._points.append([x, y])
            self._labels.append(0)
        elif event.button == 2:  # Middle click = confirm
            self._confirmed = True
            plt.close()
            return

        self._update_segmentation()

    def _on_key(self, event):
        """Handle keyboard events."""
        if event.key == 'enter':
            self._confirmed = True
            plt.close()
        elif event.key == 'r':
            self._points = []
            self._labels = []
            self._current_mask = None
            self._current_logits = None
            self._update_display()
        elif event.key == 'q':
            self._cancelled = True
            plt.close()

    def _update_segmentation(self):
        """Run SAM2 with current points and update display."""
        if not self._points:
            return

        point_coords = np.array(self._points)
        point_labels = np.array(self._labels)

        if self._current_logits is not None:
            # Refine existing mask
            masks, scores, logits = self.segmenter.refine_mask(
                point_coords, point_labels,
                mask_input=self._current_logits[None, :, :],
            )
        else:
            # First segmentation
            masks, scores, logits = self.segmenter.segment_with_points(
                point_coords, point_labels, multimask_output=True,
            )

        # Pick best mask
        best_idx = np.argmax(scores)
        self._current_mask = masks[best_idx]
        self._current_logits = logits[best_idx]

        self._update_display()

    def _update_display(self):
        """Redraw the image with current mask and points."""
        self._ax.clear()

        if self._current_mask is not None:
            overlay = apply_mask_overlay(
                self._image, self._current_mask,
                config.MASK_COLOR, config.MASK_ALPHA,
            )
            self._ax.imshow(overlay)
        else:
            self._ax.imshow(self._image)

        # Draw points
        for (x, y), label in zip(self._points, self._labels):
            color = 'lime' if label == 1 else 'red'
            marker = '+' if label == 1 else 'x'
            self._ax.plot(x, y, marker, color=color, markersize=15, markeredgewidth=3)

        # Show mask area info
        info = ""
        if self._current_mask is not None:
            area = self._current_mask.sum()
            pct = 100 * area / (self._image.shape[0] * self._image.shape[1])
            info = f" | Mask: {area:,} px ({pct:.1f}%)"

        self._ax.set_title(
            f"Points: {len(self._points)}{info} | "
            f"Enter=confirm, R=reset, Q=quit",
            fontsize=9,
        )
        self._ax.axis('off')
        self._ax.figure.canvas.draw_idle()
