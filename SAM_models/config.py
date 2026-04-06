"""Central model identifiers and inference defaults."""

from __future__ import annotations

# Hugging Face checkpoints (download on first use)
SAM1_MODEL_ID: str = "facebook/sam-vit-huge"
SAM2_LARGE_MODEL_ID: str = "facebook/sam2.1-hiera-large"
SAM3_MODEL_ID: str = "facebook/sam3"

# Published VRAM guidance for SAM 2 Large (float16/bfloat16, typical resolutions)
SAM2_LARGE_IMAGE_PEAK_MIB_ESTIMATE: float = 900.0
SAM2_LARGE_VIDEO_STEADY_MIB_ESTIMATE: float = 1800.0

# Safety margin when comparing to free VRAM (driver + fragmentation)
VRAM_HEADROOM_MIB: float = 1500.0
