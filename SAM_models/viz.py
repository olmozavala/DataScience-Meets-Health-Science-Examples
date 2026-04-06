"""Overlay segmentation masks on RGB images for display."""

from __future__ import annotations

import numpy as np
from PIL import Image


def overlay_masks(
    image_rgb: Image.Image,
    masks: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """
    Blend one or more binary/float masks onto an RGB image.

    Args:
        image_rgb: Base image in RGB mode.
        masks: Array shaped (N, H, W) or (H, W); values in [0, 1] or boolean.
        alpha: Opacity of the mask color.

    Returns:
        Composited PIL image (RGB).
    """
    base = np.asarray(image_rgb.convert("RGB"), dtype=np.float32) / 255.0
    h, w = base.shape[:2]

    if masks.ndim == 2:
        masks = masks[None, ...]
    if masks.shape[0] == 0:
        return image_rgb.copy()

    colors = np.array(
        [
            [0.95, 0.35, 0.25],
            [0.25, 0.55, 0.95],
            [0.35, 0.85, 0.45],
            [0.95, 0.85, 0.25],
            [0.75, 0.35, 0.85],
            [0.25, 0.85, 0.85],
        ],
        dtype=np.float32,
    )

    out = base.copy()
    for i in range(masks.shape[0]):
        m = masks[i].astype(np.float32)
        if m.shape != (h, w):
            # Resize mask with nearest neighbor
            pil_m = Image.fromarray((m > 0.5).astype(np.uint8) * 255, mode="L")
            pil_m = pil_m.resize((w, h), resample=Image.Resampling.NEAREST)
            m = np.asarray(pil_m, dtype=np.float32) / 255.0
        m = np.clip(m, 0.0, 1.0)
        color = colors[i % len(colors)]
        rgb = m[..., None] * color
        out = out * (1.0 - alpha * m[..., None]) + alpha * rgb
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="RGB")
