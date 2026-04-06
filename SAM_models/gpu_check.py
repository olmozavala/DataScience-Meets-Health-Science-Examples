"""GPU memory discovery and coarse fit checks for SAM 2 Large."""

from __future__ import annotations

import math
import subprocess
from dataclasses import dataclass
from typing import Any

import torch

from config import (
    SAM2_LARGE_IMAGE_PEAK_MIB_ESTIMATE,
    SAM2_LARGE_VIDEO_STEADY_MIB_ESTIMATE,
    VRAM_HEADROOM_MIB,
)


@dataclass(frozen=True)
class GpuInfo:
    """One CUDA device summary."""

    index: int
    name: str
    total_mib: float


def _parse_nvidia_smi_mib() -> list[GpuInfo]:
    """Return per-GPU total memory using nvidia-smi (MiB)."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    gpus: list[GpuInfo] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        idx, name, total = int(parts[0]), parts[1], float(parts[2])
        gpus.append(GpuInfo(index=idx, name=name, total_mib=total))
    return gpus


def list_cuda_gpus() -> list[GpuInfo]:
    """List CUDA GPUs with total VRAM; empty if unavailable."""
    if not torch.cuda.is_available():
        return []
    nvidia = _parse_nvidia_smi_mib()
    if nvidia:
        return nvidia
    # Fallback: torch only gives device count
    return [
        GpuInfo(index=i, name=torch.cuda.get_device_name(i), total_mib=float("nan"))
        for i in range(torch.cuda.device_count())
    ]


def sam2_large_fits_any_gpu() -> tuple[bool, str]:
    """
    Heuristic: whether SAM 2 Large is expected to fit on at least one GPU.

    Uses published order-of-magnitude VRAM for image/video and reserves headroom.
    """
    gpus = list_cuda_gpus()
    if not gpus:
        return False, "CUDA is not available; SAM 2 Large needs a GPU for practical use."
    # Worst-case: treat video steady-state as the bar (still conservative vs 20GB cards)
    need_mib = SAM2_LARGE_VIDEO_STEADY_MIB_ESTIMATE + VRAM_HEADROOM_MIB
    for g in gpus:
        if not math.isnan(g.total_mib) and g.total_mib >= need_mib:
            return True, (
                f"GPU {g.index} ({g.name}): ~{g.total_mib:.0f} MiB total ≥ "
                f"~{need_mib:.0f} MiB estimated for SAM 2 Large video + headroom."
            )
    return False, (
        f"No GPU reports enough total VRAM for the conservative estimate "
        f"(need ~{need_mib:.0f} MiB including headroom). "
        f"Image-only peak is lower (~{SAM2_LARGE_IMAGE_PEAK_MIB_ESTIMATE:.0f} MiB + headroom)."
    )


def gpu_summary_markdown() -> str:
    """Human-readable GPU section for the lab sidebar (Dash / other UIs)."""
    lines: list[str] = []
    gpus = list_cuda_gpus()
    if not gpus:
        lines.append("**CUDA:** not available (`torch.cuda.is_available()` is False).")
        return "\n".join(lines)
    lines.append("**CUDA devices**")
    for g in gpus:
        if not math.isnan(g.total_mib):
            lines.append(f"- `{g.index}` **{g.name}** — {g.total_mib:.0f} MiB total")
        else:
            lines.append(f"- `{g.index}` **{g.name}**")
    ok, note = sam2_large_fits_any_gpu()
    lines.append("")
    lines.append("**SAM 2 Large (heuristic)**")
    lines.append(f"- Typical image peak ≈ {SAM2_LARGE_IMAGE_PEAK_MIB_ESTIMATE:.0f} MiB")
    lines.append(f"- Typical video steady ≈ {SAM2_LARGE_VIDEO_STEADY_MIB_ESTIMATE:.0f} MiB")
    lines.append(f"- Status: **{'OK' if ok else 'check manually'}** — {note}")
    return "\n".join(lines)


def torch_cuda_mem_snapshot() -> dict[str, Any]:
    """Return basic torch CUDA memory stats for the current device, if any."""
    if not torch.cuda.is_available():
        return {"available": False}
    d = torch.cuda.current_device()
    return {
        "available": True,
        "device": d,
        "name": torch.cuda.get_device_name(d),
        "allocated_mib": torch.cuda.memory_allocated(d) / (1024**2),
        "reserved_mib": torch.cuda.memory_reserved(d) / (1024**2),
    }


def main() -> None:
    """Print GPU summary and SAM 2 Large fit heuristic (CLI)."""
    print(gpu_summary_markdown().replace("**", "").replace("`", ""))


if __name__ == "__main__":
    main()
