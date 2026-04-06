"""Load video frames for model inputs."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def load_video_frames_from_bytes(
    data: bytes,
    suffix: str = ".mp4",
    max_frames: int = 300,
    stride: int = 1,
) -> list[Image.Image]:
    """
    Decode a video from bytes into a list of RGB PIL images.

    Args:
        data: Raw file bytes (e.g. uploaded video).
        suffix: Temp file suffix for ffmpeg/OpenCV probing.
        max_frames: Hard cap to keep memory reasonable in the UI.
        stride: Use every `stride` frame (1 = all frames up to max_frames).

    Returns:
        List of RGB `PIL.Image` frames.
    """
    import cv2

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        path = Path(tmp.name)
    try:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError("Could not open video file.")
        frames: list[Image.Image] = []
        index = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if index % stride == 0:
                rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(np.asarray(rgb)))
                if len(frames) >= max_frames:
                    break
            index += 1
        cap.release()
    finally:
        path.unlink(missing_ok=True)
    return frames
