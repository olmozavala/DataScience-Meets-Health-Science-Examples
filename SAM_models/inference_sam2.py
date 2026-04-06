"""SAM 2.1 Large image and video segmentation via Transformers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from PIL import Image
from transformers import Sam2Model, Sam2Processor, Sam2VideoModel, Sam2VideoProcessor

from config import SAM2_LARGE_MODEL_ID


PromptMode = Literal["point", "box"]


@dataclass
class Sam2ImageBundle:
    """SAM 2 image model + processor."""

    model: Sam2Model
    processor: Sam2Processor
    device: torch.device
    dtype: torch.dtype


@dataclass
class Sam2VideoBundle:
    """SAM 2 video model + processor."""

    model: Sam2VideoModel
    processor: Sam2VideoProcessor
    device: torch.device
    dtype: torch.dtype


def _pick_dtype(device: torch.device) -> torch.dtype:
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def load_sam2_image(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Sam2ImageBundle:
    """
    Load SAM 2.1 Hiera Large for image segmentation.

    Args:
        device: Inference device.
        dtype: Model dtype.

    Returns:
        Ready-to-run image bundle.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = _pick_dtype(device)
    model = Sam2Model.from_pretrained(SAM2_LARGE_MODEL_ID).to(device=device, dtype=dtype)
    processor = Sam2Processor.from_pretrained(SAM2_LARGE_MODEL_ID)
    model.eval()
    return Sam2ImageBundle(model=model, processor=processor, device=device, dtype=dtype)


def load_sam2_video(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> Sam2VideoBundle:
    """
    Load SAM 2.1 Hiera Large for video tracking (same checkpoint family as image).

    Args:
        device: Inference device.
        dtype: Model dtype (bfloat16 recommended on GPU).

    Returns:
        Ready-to-run video bundle.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dtype is None:
        dtype = _pick_dtype(device)
    model = Sam2VideoModel.from_pretrained(SAM2_LARGE_MODEL_ID).to(device=device, dtype=dtype)
    processor = Sam2VideoProcessor.from_pretrained(SAM2_LARGE_MODEL_ID)
    model.eval()
    return Sam2VideoBundle(model=model, processor=processor, device=device, dtype=dtype)


def segment_image(
    bundle: Sam2ImageBundle,
    image: Image.Image,
    mode: PromptMode,
    *,
    points_xy: list[tuple[float, float]] | None = None,
    point_labels: list[int] | None = None,
    box_xyxy: tuple[float, float, float, float] | None = None,
    multimask_output: bool = False,
) -> tuple[np.ndarray, Image.Image]:
    """
    Segment one image with SAM 2 using points or a box.

    SAM 2 expects nested lists: batch -> object -> points -> (x, y).
    """
    rgb = image.convert("RGB")
    model, processor = bundle.model, bundle.processor
    dev = bundle.device

    if mode == "point":
        if not points_xy or point_labels is None or len(points_xy) != len(point_labels):
            raise ValueError("Point mode requires matching points_xy and point_labels.")
        input_points = [[[[float(x), float(y)] for x, y in points_xy]]]
        input_labels = [[list(point_labels)]]
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
        inputs = processor(images=rgb, input_boxes=input_boxes, return_tensors="pt").to(dev)

    with torch.no_grad():
        outputs = model(**inputs, multimask_output=multimask_output)

    # Returns one tensor per image: (num_masks, H, W) at original resolution
    pp = processor.post_process_masks(
        outputs.pred_masks.cpu().float(),
        inputs["original_sizes"],
    )
    m = pp[0].float()
    if m.dim() == 3 and m.shape[0] > 1 and hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
        scores = outputs.iou_scores.float().cpu()
        best = int(torch.argmax(scores[0]).item()) if scores.numel() else 0
        mask = m[best].numpy()
    else:
        mask = m[0].numpy()

    mask_np = np.asarray(mask)
    if mask_np.dtype == np.bool_:
        mask_f = mask_np.astype(np.float32)
    else:
        mask_f = (mask_np > 0.5).astype(np.float32)
    from viz import overlay_masks

    preview = overlay_masks(rgb, mask_f[None, ...])
    return mask_f, preview


def track_video_with_points(
    bundle: Sam2VideoBundle,
    frames: list[Image.Image],
    points_xy: list[tuple[float, float]],
    point_labels: list[int],
    obj_id: int = 1,
) -> tuple[list[np.ndarray], list[Image.Image]]:
    """
    Track multi-point prompts on frame 0 through the video (same nesting as image SAM 2).

    Args:
        bundle: Video model bundle.
        frames: RGB frames.
        points_xy: Pixel coordinates on frame 0 (one or more points).
        point_labels: Parallel labels (1 foreground, 0 background); length must match ``points_xy``.
        obj_id: Object id for the session.

    Returns:
        Per-frame binary masks (list) and overlay previews (list).
    """
    if not frames:
        raise ValueError("frames must be non-empty.")
    if not points_xy or point_labels is None or len(points_xy) != len(point_labels):
        raise ValueError("points_xy and point_labels must be non-empty and equal length.")
    proc, model = bundle.processor, bundle.model
    dev = bundle.device
    dtype = bundle.dtype

    inference_session = proc.init_video_session(
        video=frames,
        inference_device=dev,
        dtype=dtype,
    )
    # Match ``segment_image`` point nesting: batch -> object -> points -> (x, y).
    points = [[[[float(x), float(y)] for x, y in points_xy]]]
    labels = [[list(point_labels)]]
    proc.add_inputs_to_inference_session(
        inference_session=inference_session,
        frame_idx=0,
        obj_ids=obj_id,
        input_points=points,
        input_labels=labels,
    )

    masks_by_frame: dict[int, np.ndarray] = {}
    with torch.no_grad():
        first_out = model(inference_session=inference_session, frame_idx=0)
        pm0 = proc.post_process_masks(
            [first_out.pred_masks],
            original_sizes=[
                [inference_session.video_height, inference_session.video_width]
            ],
            binarize=True,
        )[0]
        masks_by_frame[0] = (pm0[0, 0].float().cpu().numpy() > 0.5).astype(np.float32)

        for sam_out in model.propagate_in_video_iterator(inference_session):
            pm = proc.post_process_masks(
                [sam_out.pred_masks],
                original_sizes=[
                    [inference_session.video_height, inference_session.video_width]
                ],
                binarize=True,
            )[0]
            idx = sam_out.frame_idx
            m = pm[0, 0].float().cpu().numpy()
            masks_by_frame[idx] = (m > 0.5).astype(np.float32)

    h, w = frames[0].size[1], frames[0].size[0]
    ordered = [masks_by_frame.get(i, np.zeros((h, w), dtype=np.float32)) for i in range(len(frames))]

    from viz import overlay_masks

    previews = [
        overlay_masks(frames[i].convert("RGB"), ordered[i][None, ...]) for i in range(len(frames))
    ]
    return ordered, previews


def track_video_with_point(
    bundle: Sam2VideoBundle,
    frames: list[Image.Image],
    point_xy: tuple[float, float],
    label: int = 1,
    obj_id: int = 1,
) -> tuple[list[np.ndarray], list[Image.Image]]:
    """
    Track a single point on frame 0 (convenience wrapper around :func:`track_video_with_points`).
    """
    return track_video_with_points(bundle, frames, [point_xy], [label], obj_id=obj_id)
