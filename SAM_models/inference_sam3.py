"""SAM 3 concept segmentation (text / boxes) for images and video."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor, Sam3VideoModel, Sam3VideoProcessor

from config import SAM3_MODEL_ID


@dataclass
class Sam3ImageBundle:
    """SAM 3 image detector + processor."""

    model: Sam3Model
    processor: Sam3Processor
    device: torch.device
    dtype: torch.dtype


@dataclass
class Sam3VideoBundle:
    """SAM 3 video model + processor."""

    model: Sam3VideoModel
    processor: Sam3VideoProcessor
    device: torch.device
    dtype: torch.dtype


def _pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_sam3_image(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Sam3ImageBundle:
    """Load SAM 3 for image PCS."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = _pick_dtype(device)
    model = Sam3Model.from_pretrained(SAM3_MODEL_ID).to(device=device, dtype=dtype)
    processor = Sam3Processor.from_pretrained(SAM3_MODEL_ID)
    model.eval()
    return Sam3ImageBundle(model=model, processor=processor, device=device, dtype=dtype)


def load_sam3_video(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Sam3VideoBundle:
    """Load SAM 3 video (detector + tracker)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = _pick_dtype(device)
    model = Sam3VideoModel.from_pretrained(SAM3_MODEL_ID).to(device=device, dtype=dtype)
    processor = Sam3VideoProcessor.from_pretrained(SAM3_MODEL_ID)
    model.eval()
    return Sam3VideoBundle(model=model, processor=processor, device=device, dtype=dtype)


def segment_image_text(
    bundle: Sam3ImageBundle,
    image: Image.Image,
    text: str,
    *,
    box_xyxy: tuple[float, float, float, float] | None = None,
    box_label: int = 1,
    threshold: float = 0.5,
    mask_threshold: float = 0.5,
) -> tuple[list[np.ndarray], Image.Image]:
    """
    Run text (and optional box) prompts on one image.

    Args:
        bundle: Loaded SAM 3 image bundle.
        image: RGB input.
        text: Noun phrase, e.g. ``yellow school bus``.
        box_xyxy: Optional xyxy box in pixel coordinates.
        box_label: 1 positive, 0 negative (when box is set).
        threshold: Detection score threshold for post-processing.
        mask_threshold: Mask binarization threshold.

    Returns:
        List of instance masks and a composite preview image.
    """
    rgb = image.convert("RGB")
    proc, model = bundle.processor, bundle.model
    dev = bundle.device

    if box_xyxy is not None:
        x1, y1, x2, y2 = box_xyxy
        inputs = proc(
            images=rgb,
            text=text,
            input_boxes=[[[x1, y1, x2, y2]]],
            input_boxes_labels=[[box_label]],
            return_tensors="pt",
        ).to(dev)
    else:
        inputs = proc(images=rgb, text=text, return_tensors="pt").to(dev)

    with torch.no_grad():
        outputs = model(**inputs)

    target_h, target_w = rgb.size[1], rgb.size[0]
    results0 = proc.post_process_instance_segmentation(
        outputs,
        threshold=threshold,
        mask_threshold=mask_threshold,
        target_sizes=[(target_h, target_w)],
    )[0]

    masks_t = results0["masks"]
    masks_list: list[np.ndarray] = []
    if masks_t is not None and masks_t.numel() > 0:
        for i in range(masks_t.shape[0]):
            mm = masks_t[i].float().cpu().numpy()
            masks_list.append((mm > 0.5).astype(np.float32))

    stacked = np.stack(masks_list, axis=0) if masks_list else np.zeros((0, 1, 1), dtype=np.float32)
    from viz import overlay_masks

    preview = overlay_masks(rgb, stacked) if masks_list else rgb.copy()
    return masks_list, preview


def track_video_text(
    bundle: Sam3VideoBundle,
    frames: list[Image.Image],
    text: str,
    *,
    max_frames: int = 80,
) -> tuple[dict[int, dict[str, Any]], Image.Image]:
    """
    Text-prompted video PCS: propagate and return per-frame postprocessed dicts.

    Args:
        bundle: SAM 3 video bundle.
        frames: RGB frames.
        text: Concept prompt.
        max_frames: Limit propagation length for UI responsiveness.

    Returns:
        Mapping frame_idx -> postprocessed outputs, and preview of first frame.
    """
    if not frames:
        raise ValueError("frames must be non-empty.")
    proc, model = bundle.processor, bundle.model
    dev = bundle.device
    dtype = bundle.dtype

    inference_session = proc.init_video_session(
        video=frames,
        inference_device=dev,
        processing_device="cpu",
        video_storage_device="cpu",
        dtype=dtype,
    )
    inference_session = proc.add_text_prompt(inference_session=inference_session, text=text)

    outputs_per_frame: dict[int, dict[str, Any]] = {}
    with torch.no_grad():
        for model_outputs in model.propagate_in_video_iterator(
            inference_session=inference_session,
            max_frame_num_to_track=max_frames,
        ):
            processed = proc.postprocess_outputs(inference_session, model_outputs)
            outputs_per_frame[model_outputs.frame_idx] = processed

    first = outputs_per_frame.get(0, {})
    preview = frames[0].convert("RGB")
    if "masks" in first and first["masks"] is not None:
        m = first["masks"]
        if hasattr(m, "cpu"):
            arr = m.float().cpu().numpy()
        else:
            arr = np.asarray(m, dtype=np.float32)
        if arr.ndim == 3:
            from viz import overlay_masks

            preview = overlay_masks(preview, arr)
    return outputs_per_frame, preview
