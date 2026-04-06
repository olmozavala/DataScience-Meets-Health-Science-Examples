# Segment Anything Lab

This repository contains an interactive **Dash** + **Dash Bootstrap Components** application for experimenting with Meta's family of Segment Anything Models (SAM), running local inference via Hugging Face Transformers.

## Features

The lab features a unified UI divided into three main tabs for the respective models:

### 1. SAM 1
- **Model**: `facebook/sam-vit-huge`
- **Capabilities**: Image-only segmentation based on points or bounding box prompts. Click on the Plotly image to add positive/negative points.

### 2. SAM 2.1 Large
- **Model**: `facebook/sam2.1-hiera-large`
- **Capabilities**: Image and video processing.
  - **Images**: Segments images given point or box prompts.
  - **Videos**: Object propagation and tracking across frames from a point on frame 0.

### 3. SAM 3
- **Model**: `facebook/sam3`
- **Capabilities**: Text-based prompt processing.
  - **Images**: Zero-shot concept segmentation via text (optional bounding box).
  - **Videos**: Text tracking across frames.

## Structure

The logic is modularized for clear separation of model code vs. UI:

- `app.py`: Dash entry point (Bootstrap layout, Plotly `go.Image` + `clickData` for point picking, callbacks).
- `config.py`: Model identifiers and runtime configurations, including VRAM heuristics.
- `gpu_check.py`: GPU memory discovery and SAM 2 Large fit heuristics.
- `inference_sam*.py`: Wrappers for Hugging Face inference per model family.
- `video_io.py`: Video bytes → PIL frame sequences.
- `viz.py`: Mask overlay helpers.
- `pyproject.toml`: Project metadata and dependencies (Dash, dbc, Plotly, PyTorch, etc.).

## Usage

Install UI dependencies (Dash, Plotly, etc.). PyTorch / `transformers` are usually installed in your existing ML environment (for example a dedicated `uv` env); if you want a self-contained venv:

```bash
cd segment_anything_lab
uv sync
uv sync --extra inference   # adds torch, transformers, accelerate, safetensors
```

Or, with your environment already activated: `uv pip install dash dash-bootstrap-components plotly`.

Run the app (default port **8050**):

```bash
uv run python app.py
```

Then open http://127.0.0.1:8050 in your browser (see `main()` in `app.py` if you change the port).
