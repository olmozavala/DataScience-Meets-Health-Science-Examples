"""SAM 1 (ViT-Huge) image segmentation via Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image
from transformers import SamModel, SamProcessor

from config import SAM1_MODEL_ID


PromptMode = Literal["point", "box"]


@dataclass
class Sam1Bundle:
    """Loaded SAM 1 model and processor."""

    model: SamModel
    processor: SamProcessor
    device: torch.device
    dtype: torch.dtype


def load_sam1(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Sam1Bundle:
    """
    Load `facebook/sam-vit-huge` weights and processor.

    Args:
        device: Target device; defaults to first CUDA GPU or CPU.
        dtype: Model dtype; defaults to float32 on CPU, bfloat16 on CUDA if supported.

    Returns:
        Bundle ready for `segment_image`.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        # float32 avoids bf16 edge cases on some GPUs / driver stacks with SAM ViT-huge.
        # Set SAM1_FORCE_BF16=1 to use bfloat16 on CUDA when supported (lower VRAM).
        import os

        force_bf16 = os.environ.get("SAM1_FORCE_BF16", "").lower() in ("1", "true", "yes")
        dtype = (
            torch.bfloat16
            if force_bf16 and device.type == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
    model = SamModel.from_pretrained(SAM1_MODEL_ID).to(device=device, dtype=dtype)
    processor = SamProcessor.from_pretrained(SAM1_MODEL_ID)
    model.eval()
    return Sam1Bundle(model=model, processor=processor, device=device, dtype=dtype)


def segment_image(
    bundle: Sam1Bundle,
    image: Image.Image,
    mode: PromptMode,
    *,
    points_xy: list[tuple[float, float]] | None = None,
    point_labels: list[int] | None = None,
    box_xyxy: tuple[float, float, float, float] | None = None,
    multimask_output: bool = True,
) -> tuple[np.ndarray, Image.Image]:
    """
    Run SAM 1 on one RGB image with point or box prompts.

    Args:
        bundle: Loaded SAM 1 bundle.
        image: RGB image.
        mode: ``point`` or ``box``.
        points_xy: List of (x, y) in pixel coordinates (image space).
        point_labels: 1 = positive, 0 = negative; same length as ``points_xy``.
        box_xyxy: Optional ``(x1, y1, x2, y2)`` in pixel coordinates.
        multimask_output: If True, return all mask hypotheses; best is selected for overlay.

    Returns:
        Tuple of ``mask_hw`` (float32 in {0,1}) and RGB ``Image`` overlay preview.
    """
    rgb = image.convert("RGB")
    model, processor = bundle.model, bundle.processor
    dev = bundle.device

    if mode == "point":
        if not points_xy or point_labels is None or len(points_xy) != len(point_labels):
            raise ValueError("Point mode requires matching points_xy and point_labels.")
        labels_int = [int(x) for x in point_labels]
        input_points = [[list(p) for p in points_xy]]
        input_labels = [labels_int]
        inputs = processor(
            images=rgb,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(dev)
    else:
        if box_xyxy is None:
            raise ValueError("Box mode requires box_xyxy.")
        x1, y1, x2, y2 = box_xyxy
        input_boxes = [[[x1, y1, x2, y2]]]
        inputs = processor(
            images=rgb,
            input_boxes=input_boxes,
            return_tensors="pt",
        ).to(dev)

    with torch.no_grad():
        outputs = model(**inputs)

    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu().float(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu(),
    )[0]

    scores = outputs.iou_scores.float().cpu()
    # IOU scores shape differs across transformers versions, e.g. [1,1,3] vs [1,3].
    scores_1d = scores.reshape(-1)

    if multimask_output:
        if masks.dim() == 4:
            num_masks = masks.shape[1]
        elif masks.dim() == 3:
            num_masks = masks.shape[0]
        else:
            raise ValueError(f"Unexpected post_process_masks rank: {masks.dim()}, shape={masks.shape}")

        if scores_1d.numel() < num_masks:
            raise ValueError(
                f"IoU score count ({scores_1d.numel()}) < mask count ({num_masks}); "
                f"scores.shape={tuple(scores.shape)}, masks.shape={tuple(masks.shape)}"
            )
        best = int(torch.argmax(scores_1d[:num_masks]).item())
        if masks.dim() == 4:
            mask = masks[0, best].detach().cpu().numpy()
        else:
            mask = masks[best].detach().cpu().numpy()
    else:
        if masks.dim() == 4:
            mask = masks[0, 0].detach().cpu().numpy()
        else:
            mask = masks[0].detach().cpu().numpy()

    mask_f = (mask > 0.5).astype(np.float32)
    from viz import overlay_masks

    preview = overlay_masks(rgb, mask_f[None, ...])
    return mask_f, preview
