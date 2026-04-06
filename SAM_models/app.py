"""
Dash UI: SAM 1 (points/boxes), SAM 2 Large (points/boxes, image + video),
SAM 3 (text + optional boxes; image + video).

Run::

    python app.py

Interactive points use Plotly ``go.Image`` + ``clickData`` (replaces
``streamlit_image_coordinates``).
"""

from __future__ import annotations

import base64
import io
import json
import traceback
from pathlib import Path
from typing import Any

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, dcc, html, no_update
from dash.exceptions import PreventUpdate
from PIL import Image

from gpu_check import gpu_summary_markdown
from inference_sam1 import load_sam1, segment_image as sam1_segment
from inference_sam2 import (
    load_sam2_image,
    load_sam2_video,
    segment_image as sam2_segment_image,
    track_video_with_point,
)
from inference_sam3 import (
    load_sam3_image,
    load_sam3_video,
    segment_image_text,
    track_video_text,
)
from video_io import load_video_frames_from_bytes

# --- Server-side caches (single-user local lab) ---

_device_str: str = "cuda:0"

_sam1_bundle: Any = None
_sam2_img_bundle: Any = None
_sam2_vid_bundle: Any = None
_sam3_img_bundle: Any = None
_sam3_vid_bundle: Any = None

_img_sam1: Image.Image | None = None
_img_sam2: Image.Image | None = None
_raw_sam2_vid: bytes | None = None
_frames_sam2: list[Image.Image] | None = None
_img_sam3: Image.Image | None = None
_raw_sam3_vid: bytes | None = None
_frames_sam3: list[Image.Image] | None = None

_sam2_vid_previews: list[Image.Image] | None = None
_sam2_vid_track_key: tuple[Any, ...] | None = None
_sam3_vid_result: tuple[Any, Any] | None = None
_sam3_vid_result_key: tuple[Any, ...] | None = None


def _parse_point_lines(text: str) -> tuple[list[tuple[float, float]], list[int]]:
    """
    Parse lines like ``120,80,1`` into point coordinates and labels.

    Args:
        text: Multi-line text; each line is ``x,y`` or ``x,y,label``.

    Returns:
        Parallel lists of ``(x, y)`` and integer labels (default label ``1``).
    """
    pts: list[tuple[float, float]] = []
    labs: list[int] = []
    for line in text.strip().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.replace(";", ",").split(",")]
        if len(parts) < 2:
            continue
        x, y = float(parts[0]), float(parts[1])
        lab = int(parts[2]) if len(parts) > 2 else 1
        pts.append((x, y))
        labs.append(lab)
    return pts, labs


def _decode_upload(contents: str | None) -> Image.Image | None:
    """
    Decode a Dash ``dcc.Upload`` contents string to an RGB PIL image.

    Args:
        contents: Base64 data URL or None.

    Returns:
        RGB ``PIL.Image`` or None if missing/invalid.
    """
    if not contents or "," not in contents:
        return None
    try:
        _, b64 = contents.split(",", 1)
        raw = base64.b64decode(b64)
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except (ValueError, OSError):
        return None


def _decode_upload_bytes(contents: str | None) -> bytes | None:
    """
    Decode a Dash ``dcc.Upload`` contents string to raw bytes.

    Args:
        contents: Base64 data URL or None.

    Returns:
        Raw file bytes or None.
    """
    if not contents or "," not in contents:
        return None
    try:
        _, b64 = contents.split(",", 1)
        return base64.b64decode(b64)
    except ValueError:
        return None


def _pil_to_src(img: Image.Image) -> str:
    """
    Encode a PIL image as a PNG data URL for ``html.Img``.

    Args:
        img: RGB image.

    Returns:
        Data URL string.
    """
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _pil_to_click_figure(img: Image.Image, max_h: int = 560, uirevision: str = "sam1-image") -> go.Figure:
    """
    Build a Plotly figure with a single ``go.Image`` trace for click picking.

    Args:
        img: RGB image.
        max_h: Maximum display height in pixels.
        uirevision: Plotly ``uirevision`` string to preserve zoom/pan where applicable.

    Returns:
        Figure with pixel-aligned axes for ``clickData``.
    """
    return _sam1_figure_with_point_markers(img, [], max_h=max_h, uirevision=uirevision)


def _sam1_figure_with_point_markers(
    img: Image.Image,
    points: list[list[float]],
    max_h: int = 560,
    uirevision: str = "sam1-image",
) -> go.Figure:
    """
    Build a SAM 1 figure with the image and optional point overlays.

    Positive points (label 1) are drawn green; negative (label 0) red, with a
    white outline for contrast.

    Args:
        img: RGB base image.
        points: List of ``[x, y, label]`` in pixel coordinates (same as clickData).
        max_h: Maximum display height in pixels.
        uirevision: Plotly layout ``uirevision`` (keeps zoom stable across marker updates).

    Returns:
        ``Figure`` with ``go.Image`` and ``go.Scatter`` marker traces.
    """
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]
    scale = min(1.0, max_h / float(h))
    disp_h = int(h * scale)
    disp_w = int(w * scale)
    fig = go.Figure()
    fig.add_trace(go.Image(z=arr))
    if points:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        colors = ["#22c55e" if int(p[2]) == 1 else "#ef4444" for p in points]
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker={
                    "size": 14,
                    "color": colors,
                    "line": {"color": "white", "width": 2},
                    "symbol": "circle",
                },
                hovertemplate="x=%{x:.1f}<br>y=%{y:.1f}<extra></extra>",
                name="prompt",
            )
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        uirevision=uirevision,
        xaxis=dict(showgrid=False, zeroline=False, range=[0, w], visible=False, constrain="domain"),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            range=[h, 0],
            visible=False,
            scaleanchor="x",
            scaleratio=1,
        ),
        dragmode=False,
        height=disp_h,
        width=disp_w,
        paper_bgcolor="black",
        plot_bgcolor="black",
    )
    return fig


def _jsonable(x: Any) -> Any:
    """
    Convert tensors and nested structures to JSON-serializable summaries.

    Args:
        x: Arbitrary object (often containing ``torch.Tensor``).

    Returns:
        Structure safe to serialize (tensor shapes only).
    """
    import torch

    if isinstance(x, torch.Tensor):
        return {"shape": list(x.shape), "dtype": str(x.dtype)}
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x[:20]]
    return str(type(x))


def _inference_error_alert(exc: Exception) -> dbc.Alert:
    """
    Build a detailed alert for failed model inference (message + traceback + hints).

    Args:
        exc: The exception raised during inference.

    Returns:
        ``dbc.Alert`` suitable for ``sam1-pt-out`` / ``sam1-box-out``.
    """
    msg = str(exc).strip() or repr(exc)
    tb = traceback.format_exc()
    oom_hint = ""
    if "out of memory" in msg.lower() or (
        "cuda" in msg.lower() and "memory" in msg.lower()
    ):
        oom_hint = (
            " Try: pick **cpu** in the device dropdown, reload the SAM 1 model, and run again "
            "(slower but uses system RAM). Or use a smaller image."
        )
    return dbc.Alert(
        [
            html.Strong("Segmentation failed. "),
            html.P(msg, className="mb-2 small"),
            html.Details(
                [
                    html.Summary("Technical details", className="small text-muted"),
                    html.Pre(tb, className="small mb-0 mt-1", style={"whiteSpace": "pre-wrap", "maxHeight": "240px", "overflow": "auto"}),
                ],
                className="mb-0",
            ),
            html.P(oom_hint, className="small text-info mb-0 mt-2") if oom_hint else html.Span(),
        ],
        color="danger",
        className="mb-0",
    )


def _format_points_caption(points: list[list[float]]) -> str:
    """Build a short caption listing accumulated points."""
    parts = [
        f"{i + 1}. ({p[0]:.1f}, {p[1]:.1f}) — {'positive' if int(p[2]) == 1 else 'negative'}"
        for i, p in enumerate(points)
    ]
    return "Points: " + " · ".join(parts) if parts else "No points yet — click the image."


def _sam2_vid_prompt_key(
    meta: dict[str, Any] | None,
    mf: int,
    st: int,
    store: dict[str, Any] | None,
) -> tuple[Any, ...]:
    """
    Build a key that matches a video tracking run to slider / frame preview state.

    Args:
        meta: Video upload metadata (filename, suffix).
        mf: Max frames loaded.
        st: Frame stride.
        store: Point store with ``points`` as ``[x, y, label]`` rows.

    Returns:
        Hashable tuple used for ``_sam2_vid_track_key`` comparison.
    """
    vid_sig = (meta.get("filename", ""), meta.get("suffix", "")) if meta else ("", "")
    pts = (store or {}).get("points", [])
    ptuple = tuple((float(p[0]), float(p[1]), int(p[2])) for p in pts)
    return (vid_sig, mf, st, ptuple)


def _build_layout() -> dbc.Container:
    """Compose the Dash Bootstrap layout (sidebar + tabbed main area)."""
    sidebar = dbc.Col(
        [
            dbc.Card(
                [
                    dbc.CardHeader(html.Span([dbc.Badge("System", color="secondary", className="me-2"), "Device & GPU"])),
                    dbc.CardBody(
                        [
                            dcc.Dropdown(
                                id="device-dropdown",
                                options=[{"label": x, "value": x} for x in ("cuda:0", "cpu")],
                                value="cuda:0",
                                clearable=False,
                            ),
                            html.Hr(className="my-3"),
                            dcc.Markdown(gpu_summary_markdown(), className="small mb-0"),
                        ]
                    ),
                ],
                className="shadow-sm sidebar-panel mb-3",
            ),
        ],
        width=12,
        lg=3,
    )

    main = dbc.Col(
        [
            html.Div(
                [
                    html.H1("Segment Anything Lab", className="mb-1 fw-bold"),
                    html.P(
                        "SAM 1 · SAM 2.1 Large · SAM 3 — local inference via Hugging Face Transformers",
                        className="text-muted small mb-0",
                    ),
                ],
                className="app-hero",
            ),
            dbc.Tabs(
                [
                    dbc.Tab(
                        label="SAM 1",
                        tab_id="sam1",
                        children=html.Div(_layout_sam1(), className="pt-3"),
                    ),
                    dbc.Tab(
                        label="SAM 2 Large",
                        tab_id="sam2",
                        children=html.Div(_layout_sam2(), className="pt-3"),
                    ),
                    dbc.Tab(
                        label="SAM 3",
                        tab_id="sam3",
                        children=html.Div(_layout_sam3(), className="pt-3"),
                    ),
                ],
                id="main-tabs",
                active_tab="sam1",
                className="mb-2",
            ),
        ],
        width=12,
        lg=9,
    )

    return dbc.Container(dbc.Row([sidebar, main], className="g-4"), fluid=True, className="py-4 px-3")


def _layout_sam1() -> html.Div:
    """Layout for SAM 1 with stable component IDs for callbacks."""
    return html.Div(
        [
            dbc.Card(
                [
                    dbc.CardHeader(
                        [
                            html.Span("SAM 1", className="fw-semibold me-2"),
                            dbc.Badge("ViT-Huge", color="primary", className="me-1"),
                            html.Small(" Points or box · image only", className="text-muted"),
                        ]
                    ),
                    dbc.CardBody(
                        [
                            dbc.Row(
                                [
                                    dbc.Col(
                                        [
                                            html.Small("1. Image", className="text-muted d-block mb-1"),
                                            dcc.Upload(
                                                id="sam1-upload",
                                                children=html.Div(
                                                    [
                                                        html.I(className="bi bi-cloud-upload me-2"),
                                                        "Drop an image here or ",
                                                        html.A("browse", className="fw-semibold"),
                                                    ],
                                                    className="text-center",
                                                ),
                                                accept="image/png,image/jpeg,image/webp",
                                                className="upload-zone mb-3",
                                                multiple=False,
                                            ),
                                        ],
                                        md=12,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Small("2. Prompt type", className="text-muted d-block mb-1"),
                                            dbc.RadioItems(
                                                id="sam1-mode",
                                                options=[
                                                    {"label": " Points", "value": "point"},
                                                    {"label": " Box", "value": "box"},
                                                ],
                                                value="point",
                                                inline=True,
                                                className="mb-3",
                                            ),
                                        ],
                                        md=12,
                                    ),
                                    dbc.Col(
                                        [
                                            html.Small("3. Load weights", className="text-muted d-block mb-1"),
                                            dbc.Button(
                                                [html.I(className="bi bi-download me-2"), "Load SAM 1 model"],
                                                id="sam1-load",
                                                color="primary",
                                                outline=True,
                                                className="me-2",
                                            ),
                                            html.Div(id="sam1-load-msg", className="d-inline-block align-middle"),
                                        ],
                                        md=12,
                                    ),
                                ],
                                className="g-2",
                            ),
                        ]
                    ),
                ],
                className="shadow-sm mb-3",
            ),
            html.Div(
                id="sam1-point-section",
                style={"display": "block"},
                children=[
                    dbc.Card(
                        [
                            dbc.CardHeader(html.Span([dbc.Badge("Points", color="success", className="me-2"), "Prompts"])),
                            dbc.CardBody(
                                [
                                    dcc.Markdown(
                                        "Pick **positive** or **negative** for the next click, then click the image. "
                                        "**Green** = include · **Red** = exclude. Clear / Undo as needed."
                                    ),
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dbc.RadioItems(
                                                    id="sam1-next-label",
                                                    options=[
                                                        {"label": " Positive (include)", "value": 1},
                                                        {"label": " Negative (exclude)", "value": 0},
                                                    ],
                                                    value=1,
                                                    inline=True,
                                                ),
                                                width="auto",
                                            ),
                                            dbc.Col(
                                                dbc.ButtonGroup(
                                                    [
                                                        dbc.Button("Clear", id="sam1-clear", color="warning", outline=True, size="sm"),
                                                        dbc.Button("Undo", id="sam1-undo", color="secondary", outline=True, size="sm"),
                                                    ]
                                                ),
                                                width="auto",
                                                className="ms-md-auto",
                                            ),
                                        ],
                                        className="align-items-center mb-2 flex-wrap",
                                    ),
                                    html.Div(
                                        dcc.Loading(
                                            dcc.Graph(
                                                id="sam1-graph",
                                                figure=go.Figure(),
                                                config={"displayModeBar": False},
                                                style={"cursor": "crosshair"},
                                            ),
                                            type="circle",
                                        ),
                                        className="graph-panel",
                                    ),
                                    html.Div(id="sam1-points-caption", className="small text-muted mt-2 mb-2"),
                                    dbc.Button(
                                        [html.I(className="bi bi-play-fill me-2"), "Run segmentation"],
                                        id="sam1-run-pt",
                                        color="success",
                                        size="lg",
                                        className="px-4",
                                    ),
                                    html.Div(
                                        dcc.Loading(html.Div(id="sam1-pt-out"), type="circle"),
                                        className="mt-3",
                                    ),
                                ]
                            ),
                        ],
                        className="shadow-sm border-success border-opacity-25",
                    )
                ],
            ),
            html.Div(
                id="sam1-box-section",
                style={"display": "none"},
                children=[
                    dbc.Card(
                        [
                            dbc.CardHeader(html.Span([dbc.Badge("Box", color="info", className="me-2"), "xyxy"])),
                            dbc.CardBody(
                                [
                                    dbc.Row(
                                        [
                                            dbc.Col([html.Small("x1"), dbc.Input(id="sam1-x1", type="number", value=0.0)], md=3),
                                            dbc.Col([html.Small("y1"), dbc.Input(id="sam1-y1", type="number", value=0.0)], md=3),
                                            dbc.Col([html.Small("x2"), dbc.Input(id="sam1-x2", type="number", value=0.0)], md=3),
                                            dbc.Col([html.Small("y2"), dbc.Input(id="sam1-y2", type="number", value=0.0)], md=3),
                                        ],
                                        className="g-2 mb-3",
                                    ),
                                    html.Img(id="sam1-box-ref", src="", className="img-fluid rounded border mb-3"),
                                    dbc.Button(
                                        [html.I(className="bi bi-play-fill me-2"), "Run segmentation (box)"],
                                        id="sam1-run-box",
                                        color="success",
                                        size="lg",
                                    ),
                                    dcc.Loading(html.Div(id="sam1-box-out"), type="circle", className="mt-3"),
                                ]
                            ),
                        ],
                        className="shadow-sm",
                    )
                ],
            ),
            dcc.Store(id="sam1-points-store", data={"points": [], "last_click_ser": None}),
        ]
    )


def _layout_sam2() -> html.Div:
    """Layout for SAM 2."""
    return html.Div(
        [
            dcc.Markdown(
                "**SAM 2.1 Large** — points or boxes. **Image** and **video** "
                "(propagation from a point on frame 0)."
            ),
            dbc.RadioItems(
                id="sam2-media",
                options=[{"label": x, "value": x} for x in ("image", "video")],
                value="image",
                inline=True,
                className="mb-2",
            ),
            dcc.Upload(
                id="sam2-upload",
                children=html.Div(["Drag and drop or ", html.A("select a file")]),
                accept="image/png,image/jpeg,image/webp,video/mp4,video/webm,video/x-msvideo",
                className="border rounded p-2 mb-2",
                multiple=False,
            ),
            html.Div(id="sam2-image-block", style={"display": "block"}, children=_sam2_image_children()),
            html.Div(
                id="sam2-video-block",
                style={"display": "none"},
                children=[
                    dcc.Store(id="sam2-vid-meta", data=None),
                    dcc.Store(id="sam2-vid-points-store", data={"points": [], "last_ser": None}),
                    html.Label("Max frames to load"),
                    dcc.Slider(id="sam2-maxf", min=8, max=300, step=1, value=80),
                    html.Label("Frame stride"),
                    dcc.Slider(id="sam2-stride", min=1, max=5, step=1, value=1),
                    html.Div(id="sam2-vid-content"),
                    dcc.Loading(
                        html.Div(id="sam2-vid-track-out", className="mt-3"),
                        type="circle",
                        className="mt-3",
                    ),
                ],
            ),
            dcc.Store(id="sam2-im-points-store", data={"points": [], "last_ser": None}),
        ]
    )


def _sam2_image_children() -> list[Any]:
    """Children for the SAM 2 image sub-panel (stable IDs)."""
    return [
        dbc.Button("Load SAM 2 (image) model", id="sam2-im-load", color="primary", className="me-2"),
        html.Span(id="sam2-im-load-msg", className="text-success small"),
        html.Hr(),
        dbc.RadioItems(
            id="sam2-im-mode",
            options=[{"label": x, "value": x} for x in ("point", "box")],
            value="point",
            inline=True,
        ),
        html.Div(id="sam2-im-panel"),
    ]


def _layout_sam3() -> html.Div:
    """Layout for SAM 3."""
    return html.Div(
        [
            dcc.Markdown(
                "**SAM 3** — **text** concept prompts (and optional **boxes** on images). "
                "Video uses the same text prompt with tracking."
            ),
            dbc.RadioItems(
                id="sam3-media",
                options=[{"label": x, "value": x} for x in ("image", "video")],
                value="image",
                inline=True,
                className="mb-2",
            ),
            dcc.Upload(
                id="sam3-upload",
                children=html.Div(["Drag and drop or ", html.A("select a file")]),
                accept="image/png,image/jpeg,image/webp,video/mp4,video/webm,video/x-msvideo",
                className="border rounded p-2 mb-2",
                multiple=False,
            ),
            dbc.Input(id="sam3-text", type="text", value="person", placeholder="Text prompt"),
            dbc.Checklist(
                id="sam3-use-box",
                options=[{"label": " Optional box prompt (xyxy, image only)", "value": 1}],
                value=[],
                className="mb-2",
            ),
            html.Div(id="sam3-image-panel", style={"display": "block"}),
            html.Div(
                id="sam3-video-panel",
                style={"display": "none"},
                children=[
                    dcc.Store(id="sam3-vid-meta", data=None),
                    html.Label("Max frames"),
                    dcc.Slider(id="sam3-maxf", min=8, max=200, step=1, value=50),
                    html.Div(id="sam3-vid-body"),
                    html.Div(id="sam3-vid-track-out", className="mt-2"),
                ],
            ),
        ]
    )


def create_app() -> dash.Dash:
    """
    Build and configure the Dash application with Bootstrap styling.

    Returns:
        Configured ``Dash`` instance (callbacks registered).
    """
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.FLATLY, dbc.icons.BOOTSTRAP],
        suppress_callback_exceptions=True,
    )
    app.title = "Segment Anything Lab"
    app.layout = html.Div([dcc.Location(id="url"), _build_layout()])
    _register_callbacks(app)
    return app


def _register_callbacks(app: dash.Dash) -> None:
    """Attach all callbacks to the given Dash app."""

    @app.callback(
        Output("sam1-point-section", "style"),
        Output("sam1-box-section", "style"),
        Input("sam1-mode", "value"),
    )
    def _sam1_toggle_sections(mode: str | None) -> tuple[dict[str, str], dict[str, str]]:
        if mode == "box":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output("sam1-graph", "figure"),
        Output("sam1-box-ref", "src"),
        Output("sam1-x2", "value"),
        Output("sam1-y2", "value"),
        Output("sam1-points-store", "data", allow_duplicate=True),
        Input("sam1-upload", "contents"),
        State("sam1-upload", "filename"),
        prevent_initial_call=True,
    )
    def _sam1_upload(contents: str | None, filename: str | None) -> Any:
        global _img_sam1
        empty_store = {"points": [], "last_click_ser": None}
        if not contents:
            _img_sam1 = None
            return go.Figure(), "", 0.0, 0.0, empty_store
        img = _decode_upload(contents)
        if img is None:
            _img_sam1 = None
            return go.Figure(), "", 0.0, 0.0, empty_store
        _img_sam1 = img
        w, h = img.size
        fig = _sam1_figure_with_point_markers(img, [])
        return fig, _pil_to_src(img), float(w), float(h), empty_store

    @app.callback(
        Output("sam1-points-store", "data"),
        Output("sam1-points-caption", "children"),
        Output("sam1-graph", "figure", allow_duplicate=True),
        Input("sam1-graph", "clickData"),
        Input("sam1-clear", "n_clicks"),
        Input("sam1-undo", "n_clicks"),
        State("sam1-points-store", "data"),
        State("sam1-next-label", "value"),
        prevent_initial_call=True,
    )
    def _sam1_points(
        click_data: dict[str, Any] | None,
        n_clear: int | None,
        n_undo: int | None,
        store: dict[str, Any] | None,
        next_label: int | None,
    ) -> tuple[dict[str, Any], str, go.Figure]:
        global _img_sam1
        if _img_sam1 is None:
            raise PreventUpdate
        if not callback_context.triggered:
            raise PreventUpdate
        tid = callback_context.triggered[0]["prop_id"].split(".")[0]

        store = store or {"points": [], "last_click_ser": None}
        points: list[list[float]] = [list(p) for p in store.get("points", [])]

        if tid == "sam1-clear" and n_clear:
            empty: list[list[float]] = []
            return (
                {"points": [], "last_click_ser": None},
                "No points yet — click the image.",
                _sam1_figure_with_point_markers(_img_sam1, empty),
            )
        if tid == "sam1-undo" and n_undo and points:
            points.pop()
            return (
                {"points": points, "last_click_ser": None},
                _format_points_caption(points),
                _sam1_figure_with_point_markers(_img_sam1, points),
            )

        if tid != "sam1-graph" or not click_data or not click_data.get("points"):
            raise PreventUpdate

        ser = json.dumps(click_data, sort_keys=True)
        if store.get("last_click_ser") == ser:
            raise PreventUpdate

        pt0 = click_data["points"][0]
        cx, cy = float(pt0["x"]), float(pt0["y"])
        lab = int(next_label) if next_label is not None else 1
        points.append([cx, cy, float(lab)])
        return (
            {"points": points, "last_click_ser": ser},
            _format_points_caption(points),
            _sam1_figure_with_point_markers(_img_sam1, points),
        )

    @app.callback(
        Output("sam1-load-msg", "children"),
        Input("sam1-load", "n_clicks"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam1_load(n: int | None, device: str | None) -> str:
        global _sam1_bundle, _device_str
        if not n:
            raise PreventUpdate
        import torch

        if device:
            _device_str = device
        dev = torch.device(_device_str)
        _sam1_bundle = load_sam1(device=dev)
        return html.Span(
            [
                dbc.Badge("SAM 1 ready", color="success", className="me-2"),
                html.Small(f"on { _device_str }", className="text-muted"),
            ]
        )

    @app.callback(
        Output("sam1-pt-out", "children"),
        Output("sam1-graph", "figure", allow_duplicate=True),
        Input("sam1-run-pt", "n_clicks"),
        State("sam1-points-store", "data"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam1_run_pt(
        n: int | None,
        store: dict[str, Any] | None,
        device: str | None,
    ) -> tuple[Any, Any]:
        """
        Run SAM 1 point segmentation and draw the mask overlay in the main graph
        together with prompt markers; summary stats go below the graph.
        """
        global _sam1_bundle, _img_sam1, _device_str
        if not n:
            raise PreventUpdate
        if device:
            _device_str = device
        if _img_sam1 is None:
            return (
                dbc.Alert(
                    [
                        html.Strong("No image loaded. "),
                        "Use step 1 to upload an image first.",
                    ],
                    color="warning",
                    className="mb-0",
                ),
                no_update,
            )
        if _sam1_bundle is None:
            return (
                dbc.Alert(
                    [
                        html.Strong("Model not loaded. "),
                        'Click “Load SAM 1 model” in step 3 and wait for the green “SAM 1 ready” message, ',
                        "then run segmentation again.",
                    ],
                    color="info",
                    className="mb-0",
                ),
                no_update,
            )
        store = store or {}
        raw_pts = store.get("points", [])
        pts = [(float(p[0]), float(p[1])) for p in raw_pts]
        labs = [int(p[2]) for p in raw_pts]
        if not pts:
            return (
                dbc.Alert(
                    "Add at least one point on the image (green/red) before running.",
                    color="warning",
                    className="mb-0",
                ),
                no_update,
            )
        try:
            mask, preview = sam1_segment(_sam1_bundle, _img_sam1, "point", points_xy=pts, point_labels=labs)
        except Exception as exc:
            return _inference_error_alert(exc), no_update
        area = float(mask.sum())
        point_rows: list[list[float]] = [list(p) for p in raw_pts]
        fig = _sam1_figure_with_point_markers(preview, point_rows)
        summary = dbc.Alert(
            [
                html.I(className="bi bi-check-circle-fill me-2"),
                html.Strong(f"Done. "),
                f"Mask area ≈ {area:.0f} px. The colored overlay is in the image above; adjust points and run again if needed.",
            ],
            color="success",
            className="mb-0",
        )
        return summary, fig

    @app.callback(
        Output("sam1-box-out", "children"),
        Input("sam1-run-box", "n_clicks"),
        State("sam1-x1", "value"),
        State("sam1-y1", "value"),
        State("sam1-x2", "value"),
        State("sam1-y2", "value"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam1_run_box(
        n: int | None,
        x1: float | None,
        y1: float | None,
        x2: float | None,
        y2: float | None,
        device: str | None,
    ) -> Any:
        global _sam1_bundle, _img_sam1, _device_str
        if not n:
            raise PreventUpdate
        if device:
            _device_str = device
        if _img_sam1 is None:
            return dbc.Alert("Upload an image in step 1 first.", color="warning", className="mb-0")
        if _sam1_bundle is None:
            return dbc.Alert(
                'Load the SAM 1 model (step 3) before running box segmentation.',
                color="info",
                className="mb-0",
            )
        try:
            mask, preview = sam1_segment(
                _sam1_bundle,
                _img_sam1,
                "box",
                box_xyxy=(float(x1 or 0), float(y1 or 0), float(x2 or 0), float(y2 or 0)),
            )
        except Exception as exc:
            return _inference_error_alert(exc)
        return html.Img(src=_pil_to_src(preview), className="img-fluid rounded border shadow-sm")

    # --- SAM 2 visibility + image ---
    @app.callback(
        Output("sam2-image-block", "style"),
        Output("sam2-video-block", "style"),
        Input("sam2-media", "value"),
    )
    def _sam2_media_visibility(media: str | None) -> tuple[dict[str, str], dict[str, str]]:
        if media == "video":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output("sam2-im-panel", "children"),
        Output("sam2-im-points-store", "data"),
        Input("sam2-im-mode", "value"),
        Input("sam2-upload", "contents"),
        State("sam2-media", "value"),
    )
    def _sam2_im_panel(mode: str | None, contents: str | None, media: str | None) -> tuple[Any, dict[str, Any]]:
        global _img_sam2
        empty_store: dict[str, Any] = {"points": [], "last_ser": None}
        if media == "video":
            return (
                html.P(
                    "Video is handled in the section below when **Video** is selected.",
                    className="text-muted small",
                ),
                empty_store,
            )
        if not contents:
            return html.P("Upload an image first.", className="text-muted small"), empty_store
        img = _decode_upload(contents)
        if img is None:
            return dbc.Alert("Could not decode image.", color="warning"), empty_store
        _img_sam2 = img
        if mode == "point":
            return (
                html.Div(
                    [
                        dcc.Markdown(
                            "Pick **positive** or **negative** for the next click, then click the image. "
                            "**Green** = include · **Red** = exclude. Clear / Undo as needed. "
                            "Optional: extra points in the text area (`x,y,0` for negatives)."
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    dbc.RadioItems(
                                        id="sam2-im-next-label",
                                        options=[
                                            {"label": " Positive (include)", "value": 1},
                                            {"label": " Negative (exclude)", "value": 0},
                                        ],
                                        value=1,
                                        inline=True,
                                    ),
                                    width="auto",
                                ),
                                dbc.Col(
                                    dbc.ButtonGroup(
                                        [
                                            dbc.Button(
                                                "Clear",
                                                id="sam2-im-clear",
                                                color="warning",
                                                outline=True,
                                                size="sm",
                                            ),
                                            dbc.Button(
                                                "Undo",
                                                id="sam2-im-undo",
                                                color="secondary",
                                                outline=True,
                                                size="sm",
                                            ),
                                        ]
                                    ),
                                    width="auto",
                                    className="ms-md-auto",
                                ),
                            ],
                            className="align-items-center mb-2 flex-wrap",
                        ),
                        dcc.Loading(
                            dcc.Graph(
                                id="sam2-im-graph",
                                figure=_sam1_figure_with_point_markers(img, []),
                                config={"displayModeBar": False},
                                style={"cursor": "crosshair"},
                            ),
                            type="circle",
                        ),
                        html.Div(id="sam2-im-points-caption", className="small text-muted mt-2 mb-2"),
                        dbc.Textarea(
                            id="sam2-im-extra",
                            placeholder="Extra points: one x,y,label per line",
                            rows=3,
                        ),
                        dbc.Button(
                            "Run SAM 2 image (points)",
                            id="sam2-im-run-pt",
                            color="success",
                            className="mt-2",
                        ),
                        html.Div(id="sam2-im-pt-out"),
                    ]
                ),
                empty_store,
            )
        return (
            html.Div(
                [
                    dbc.Row(
                        [
                            dbc.Col(dbc.Input(id="s2x1", type="number", value=0.0), width=3),
                            dbc.Col(dbc.Input(id="s2y1", type="number", value=0.0), width=3),
                            dbc.Col(dbc.Input(id="s2x2", type="number", value=float(img.size[0])), width=3),
                            dbc.Col(dbc.Input(id="s2y2", type="number", value=float(img.size[1])), width=3),
                        ],
                        className="g-2 mb-2",
                    ),
                    html.Img(src=_pil_to_src(img), style={"maxWidth": "100%"}),
                    dbc.Button("Run SAM 2 image (box)", id="sam2-im-run-box", color="success", className="mt-2"),
                    html.Div(id="sam2-im-box-out"),
                ]
            ),
            empty_store,
        )

    @app.callback(
        Output("sam2-im-points-store", "data", allow_duplicate=True),
        Output("sam2-im-points-caption", "children"),
        Output("sam2-im-graph", "figure", allow_duplicate=True),
        Input("sam2-im-graph", "clickData"),
        Input("sam2-im-clear", "n_clicks"),
        Input("sam2-im-undo", "n_clicks"),
        State("sam2-im-points-store", "data"),
        State("sam2-im-next-label", "value"),
        prevent_initial_call=True,
    )
    def _sam2_im_points(
        click_data: dict[str, Any] | None,
        n_clear: int | None,
        n_undo: int | None,
        store: dict[str, Any] | None,
        next_label: int | None,
    ) -> tuple[dict[str, Any], str, go.Figure]:
        """
        Match SAM 1 point UX: update markers on the graph, caption, and store together.
        """
        global _img_sam2
        if _img_sam2 is None:
            raise PreventUpdate
        if not callback_context.triggered:
            raise PreventUpdate
        tid = callback_context.triggered[0]["prop_id"].split(".")[0]

        store = store or {"points": [], "last_ser": None}
        points: list[list[float]] = [list(p) for p in store.get("points", [])]

        if tid == "sam2-im-clear" and n_clear:
            empty: list[list[float]] = []
            return (
                {"points": [], "last_ser": None},
                "No points yet — click the image.",
                _sam1_figure_with_point_markers(_img_sam2, empty),
            )
        if tid == "sam2-im-undo" and n_undo and points:
            points.pop()
            return (
                {"points": points, "last_ser": None},
                _format_points_caption(points),
                _sam1_figure_with_point_markers(_img_sam2, points),
            )

        if tid != "sam2-im-graph" or not click_data or not click_data.get("points"):
            raise PreventUpdate

        ser = json.dumps(click_data, sort_keys=True)
        if store.get("last_ser") == ser:
            raise PreventUpdate

        pt0 = click_data["points"][0]
        cx, cy = float(pt0["x"]), float(pt0["y"])
        lab = int(next_label) if next_label is not None else 1
        points.append([cx, cy, float(lab)])
        return (
            {"points": points, "last_ser": ser},
            _format_points_caption(points),
            _sam1_figure_with_point_markers(_img_sam2, points),
        )

    @app.callback(
        Output("sam2-im-load-msg", "children"),
        Input("sam2-im-load", "n_clicks"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam2_im_load(n: int | None, device: str | None) -> str:
        global _sam2_img_bundle, _device_str
        if not n:
            raise PreventUpdate
        import torch

        if device:
            _device_str = device
        _sam2_img_bundle = load_sam2_image(device=torch.device(_device_str))
        return "SAM 2 image model ready."

    @app.callback(
        Output("sam2-im-pt-out", "children"),
        Output("sam2-im-graph", "figure", allow_duplicate=True),
        Input("sam2-im-run-pt", "n_clicks"),
        State("sam2-im-points-store", "data"),
        State("sam2-im-extra", "value"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam2_run_im_pt(
        n: int | None,
        store: dict[str, Any] | None,
        extra: str | None,
        device: str | None,
    ) -> tuple[Any, Any]:
        """
        Run SAM 2 point segmentation and show the mask in the main graph (like SAM 1), with a summary below.
        """
        global _sam2_img_bundle, _img_sam2, _device_str
        if not n or _sam2_img_bundle is None or _img_sam2 is None:
            raise PreventUpdate
        if device:
            _device_str = device
        store = store or {}
        point_rows: list[list[float]] = [list(p) for p in store.get("points", [])]
        pts: list[tuple[float, float]] = []
        labs: list[int] = []
        for p in store.get("points", []):
            pts.append((float(p[0]), float(p[1])))
            labs.append(int(p[2]))
        ep, el = _parse_point_lines(extra or "")
        for (x, y), lab in zip(ep, el):
            point_rows.append([float(x), float(y), float(lab)])
            pts.append((float(x), float(y)))
            labs.append(int(lab))
        if not pts:
            return (
                dbc.Alert("Need at least one point.", color="warning", className="mb-0"),
                no_update,
            )
        try:
            mask, preview = sam2_segment_image(
                _sam2_img_bundle, _img_sam2, "point", points_xy=pts, point_labels=labs
            )
        except Exception as exc:
            return _inference_error_alert(exc), no_update
        area = float(mask.sum())
        fig = _sam1_figure_with_point_markers(preview, point_rows)
        summary = dbc.Alert(
            [
                html.I(className="bi bi-check-circle-fill me-2"),
                html.Strong("Done. "),
                f"Mask area ≈ {area:.0f} px. The overlay is in the image above; adjust points and run again if needed.",
            ],
            color="success",
            className="mb-0",
        )
        return summary, fig

    @app.callback(
        Output("sam2-im-box-out", "children"),
        Input("sam2-im-run-box", "n_clicks"),
        State("s2x1", "value"),
        State("s2y1", "value"),
        State("s2x2", "value"),
        State("s2y2", "value"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam2_run_im_box(
        n: int | None,
        x1: float | None,
        y1: float | None,
        x2: float | None,
        y2: float | None,
        device: str | None,
    ) -> Any:
        global _sam2_img_bundle, _img_sam2, _device_str
        if not n or _sam2_img_bundle is None or _img_sam2 is None:
            raise PreventUpdate
        if device:
            _device_str = device
        mask, preview = sam2_segment_image(
            _sam2_img_bundle,
            _img_sam2,
            "box",
            box_xyxy=(float(x1 or 0), float(y1 or 0), float(x2 or 0), float(y2 or 0)),
        )
        return html.Img(src=_pil_to_src(preview), style={"maxWidth": "100%"})

    @app.callback(
        Output("sam2-vid-meta", "data"),
        Input("sam2-upload", "contents"),
        Input("sam2-media", "value"),
        State("sam2-upload", "filename"),
    )
    def _sam2_vid_meta(contents: str | None, media: str | None, filename: str | None) -> Any:
        global _raw_sam2_vid, _img_sam2, _frames_sam2, _sam2_vid_previews, _sam2_vid_track_key
        if media != "video" or not contents:
            _raw_sam2_vid = None
            _frames_sam2 = None
            return None
        raw = _decode_upload_bytes(contents)
        if raw is None:
            _raw_sam2_vid = None
            return None
        _raw_sam2_vid = raw
        _img_sam2 = None
        _frames_sam2 = None
        _sam2_vid_previews = None
        _sam2_vid_track_key = None
        suf = Path(filename or "clip.mp4").suffix or ".mp4"
        return {"suffix": suf, "filename": filename or ""}

    @app.callback(
        Output("sam2-vid-content", "children"),
        Output("sam2-vid-points-store", "data"),
        Input("sam2-vid-meta", "data"),
        Input("sam2-maxf", "value"),
        Input("sam2-stride", "value"),
    )
    def _sam2_vid_content(
        meta: dict[str, Any] | None, max_f: float | None, stride: float | None
    ) -> tuple[Any, dict[str, Any]]:
        global _raw_sam2_vid, _frames_sam2, _sam2_vid_previews, _sam2_vid_track_key
        empty_pts: dict[str, Any] = {"points": [], "last_ser": None}
        if not meta or _raw_sam2_vid is None:
            return html.P("Upload a video file.", className="text-muted"), empty_pts
        mf = int(max_f) if max_f is not None else 80
        st = int(stride) if stride is not None else 1
        suf = meta.get("suffix") or ".mp4"
        _frames_sam2 = load_video_frames_from_bytes(_raw_sam2_vid, suffix=suf, max_frames=mf, stride=st)
        _sam2_vid_previews = None
        _sam2_vid_track_key = None
        if not _frames_sam2:
            return dbc.Alert("No frames decoded.", color="danger"), empty_pts
        p0 = _frames_sam2[0].copy()
        mx, my = float(p0.size[0]) / 2.0, float(p0.size[1]) / 2.0
        return html.Div(
            [
                dcc.Markdown(
                    f"Loaded **{len(_frames_sam2)}** frame(s). **Frame 0** — choose positive or negative "
                    "for the next click, then click the image. **All** points are sent to the video model "
                    "(same nesting as SAM 2 image). Coordinates below show the **last** click for editing."
                ),
                dbc.RadioItems(
                    id="sam2-vid-next-label",
                    options=[
                        {"label": " Next click: positive (green)", "value": 1},
                        {"label": " Next click: negative (red)", "value": 0},
                    ],
                    value=1,
                    inline=True,
                    className="mb-2",
                ),
                dcc.Loading(
                    dcc.Graph(
                        id="sam2-vid-graph",
                        figure=go.Figure(),
                        config={"displayModeBar": False},
                    ),
                    type="circle",
                ),
                dbc.Row(
                    [
                        dbc.Col(
                            [
                                html.Label("Last click x (frame 0)"),
                                dbc.Input(id="sam2-px", type="number", value=mx),
                            ],
                            width=6,
                        ),
                        dbc.Col(
                            [
                                html.Label("Last click y (frame 0)"),
                                dbc.Input(id="sam2-py", type="number", value=my),
                            ],
                            width=6,
                        ),
                    ],
                    className="g-2",
                ),
                dbc.Button("Load SAM 2 (video) model", id="sam2-vid-load", color="primary", className="me-2 mt-2"),
                html.Span(id="sam2-vid-load-msg", className="text-success small"),
                dbc.Button("Track in video", id="sam2-vid-run", color="success", className="mt-2 d-block"),
            ]
        ), empty_pts

    @app.callback(
        Output("sam2-vid-graph", "figure"),
        Input("sam2-vid-points-store", "data"),
        prevent_initial_call=True,
    )
    def _sam2_vid_figure(points_data: dict[str, Any] | None) -> go.Figure:
        """
        Redraw frame 0 with green/red markers whenever the prompt point store changes.

        Depends on ``_frames_sam2`` set by ``_sam2_vid_content`` in the same request
        when the store is reset after a new load.
        """
        global _frames_sam2
        if _frames_sam2 is None:
            return go.Figure()
        p0 = _frames_sam2[0].copy()
        pts = (points_data or {}).get("points", [])
        return _sam1_figure_with_point_markers(p0, pts, uirevision="sam2-vid")

    @app.callback(
        Output("sam2-vid-points-store", "data", allow_duplicate=True),
        Output("sam2-px", "value"),
        Output("sam2-py", "value"),
        Input("sam2-vid-graph", "clickData"),
        State("sam2-vid-points-store", "data"),
        prevent_initial_call=True,
    )
    def _sam2_vid_click(
        click_data: dict[str, Any] | None, store: dict[str, Any] | None
    ) -> tuple[dict[str, Any], float, float]:
        """
        Record clicks on frame 0, show markers via ``_sam2_vid_figure``, and sync x/y fields.

        Args:
            click_data: Plotly ``clickData`` from the frame-0 graph.
            store: Prior ``sam2-vid-points-store`` payload.

        Returns:
            Updated store, and the latest click coordinates for numeric fields / tracking.
        """
        if not click_data or not click_data.get("points"):
            raise PreventUpdate
        ser = json.dumps(click_data, sort_keys=True)
        store = store or {"points": [], "last_ser": None}
        if store.get("last_ser") == ser:
            raise PreventUpdate
        p0 = click_data["points"][0]
        x, y = float(p0["x"]), float(p0["y"])
        pts: list[list[float]] = [list(p) for p in store.get("points", [])]
        pts.append([x, y, 1.0])
        return {"points": pts, "last_ser": ser}, x, y

    @app.callback(
        Output("sam2-vid-load-msg", "children"),
        Input("sam2-vid-load", "n_clicks"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam2_vid_load(n: int | None, device: str | None) -> str:
        global _sam2_vid_bundle, _device_str
        if not n:
            raise PreventUpdate
        import torch

        if device:
            _device_str = device
        _sam2_vid_bundle = load_sam2_video(device=torch.device(_device_str))
        return "SAM 2 video model ready."

    @app.callback(
        Output("sam2-vid-track-out", "children"),
        Input("sam2-vid-run", "n_clicks"),
        State("sam2-px", "value"),
        State("sam2-py", "value"),
        State("sam2-vid-meta", "data"),
        State("sam2-maxf", "value"),
        State("sam2-stride", "value"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam2_vid_track(
        n: int | None,
        px: float | None,
        py: float | None,
        meta: dict[str, Any] | None,
        max_f: float | None,
        stride: float | None,
        device: str | None,
    ) -> Any:
        global _sam2_vid_bundle, _frames_sam2, _sam2_vid_previews, _sam2_vid_track_key, _device_str
        if not n or _sam2_vid_bundle is None or _frames_sam2 is None or not meta:
            raise PreventUpdate
        if device:
            _device_str = device
        vid_sig = (meta.get("filename", ""), meta.get("suffix", ""))
        mf = int(max_f) if max_f is not None else 80
        st = int(stride) if stride is not None else 1
        bx, by = float(px or 0), float(py or 0)
        key = (vid_sig, mf, st, bx, by)
        masks, previews = track_video_with_point(_sam2_vid_bundle, _frames_sam2, (bx, by))
        _sam2_vid_previews = previews
        _sam2_vid_track_key = key
        return html.Div(
            [
                html.Label("Frame"),
                dcc.Slider(
                    id="sam2-vid-frame",
                    min=0,
                    max=max(0, len(previews) - 1),
                    step=1,
                    value=0,
                    marks=None,
                ),
                html.Div(id="sam2-vid-frame-img"),
            ]
        )

    @app.callback(
        Output("sam2-vid-frame-img", "children"),
        Input("sam2-vid-frame", "value"),
        State("sam2-vid-meta", "data"),
        State("sam2-maxf", "value"),
        State("sam2-stride", "value"),
        State("sam2-px", "value"),
        State("sam2-py", "value"),
    )
    def _sam2_vid_show_frame(
        idx: float | None,
        meta: dict[str, Any] | None,
        max_f: float | None,
        stride: float | None,
        px: float | None,
        py: float | None,
    ) -> Any:
        global _sam2_vid_previews, _sam2_vid_track_key
        if _sam2_vid_previews is None or not meta:
            return html.Div()
        mf = int(max_f) if max_f is not None else 80
        st = int(stride) if stride is not None else 1
        vid_sig = (meta.get("filename", ""), meta.get("suffix", ""))
        key = (vid_sig, mf, st, float(px or 0), float(py or 0))
        if _sam2_vid_track_key != key:
            return html.Div()
        i = int(idx or 0)
        i = max(0, min(i, len(_sam2_vid_previews) - 1))
        im = _sam2_vid_previews[i]
        return html.Div([html.Img(src=_pil_to_src(im), style={"maxWidth": "100%"}), html.P(f"Frame {i}", className="small")])

    # --- SAM 3 ---
    @app.callback(
        Output("sam3-image-panel", "style"),
        Output("sam3-video-panel", "style"),
        Input("sam3-media", "value"),
    )
    def _sam3_panels(media: str | None) -> tuple[dict[str, str], dict[str, str]]:
        if media == "video":
            return {"display": "none"}, {"display": "block"}
        return {"display": "block"}, {"display": "none"}

    @app.callback(
        Output("sam3-image-panel", "children"),
        Input("sam3-upload", "contents"),
        Input("sam3-use-box", "value"),
        State("sam3-upload", "filename"),
        State("sam3-text", "value"),
    )
    def _sam3_image_panel(
        contents: str | None,
        use_box: list[Any] | None,
        filename: str | None,
        text: str | None,
    ) -> Any:
        global _img_sam3
        if not contents:
            _img_sam3 = None
            return html.P("Upload an image.", className="text-muted")
        img = _decode_upload(contents)
        if img is None:
            return dbc.Alert("Could not decode image.", color="warning")
        _img_sam3 = img
        show_box = bool(use_box and 1 in use_box)
        return html.Div(
            [
                html.Div(
                    id="sam3-box-wrap",
                    style={"display": "block" if show_box else "none"},
                    children=[
                        dbc.Row(
                            [
                                dbc.Col(dbc.Input(id="s3x1", type="number", value=0.0), width=3),
                                dbc.Col(dbc.Input(id="s3y1", type="number", value=0.0), width=3),
                                dbc.Col(dbc.Input(id="s3x2", type="number", value=float(img.size[0])), width=3),
                                dbc.Col(dbc.Input(id="s3y2", type="number", value=float(img.size[1])), width=3),
                            ],
                            className="g-2 mb-2",
                        ),
                        dbc.RadioItems(
                            id="s3bl",
                            options=[{"label": "positive", "value": 1}, {"label": "negative", "value": 0}],
                            value=1,
                            inline=True,
                        ),
                    ],
                ),
                dbc.Button("Load SAM 3 (image) model", id="sam3-im-load", color="primary", className="me-2"),
                html.Span(id="sam3-im-load-msg", className="text-success small"),
                dbc.Button("Run SAM 3 (image)", id="sam3-im-run", color="success", className="mt-2 d-block"),
                html.Div(id="sam3-im-out"),
            ]
        )

    @app.callback(
        Output("sam3-im-load-msg", "children"),
        Input("sam3-im-load", "n_clicks"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam3_im_load(n: int | None, device: str | None) -> str:
        global _sam3_img_bundle, _device_str
        if not n:
            raise PreventUpdate
        import torch

        if device:
            _device_str = device
        _sam3_img_bundle = load_sam3_image(device=torch.device(_device_str))
        return "SAM 3 image model ready."

    @app.callback(
        Output("sam3-im-out", "children"),
        Input("sam3-im-run", "n_clicks"),
        State("sam3-text", "value"),
        State("sam3-use-box", "value"),
        State("s3x1", "value"),
        State("s3y1", "value"),
        State("s3x2", "value"),
        State("s3y2", "value"),
        State("s3bl", "value"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam3_im_run(
        n: int | None,
        text: str | None,
        use_box: list[Any] | None,
        x1: float | None,
        y1: float | None,
        x2: float | None,
        y2: float | None,
        bl: int | None,
        device: str | None,
    ) -> Any:
        global _sam3_img_bundle, _img_sam3, _device_str
        if not n or _sam3_img_bundle is None or _img_sam3 is None:
            raise PreventUpdate
        if device:
            _device_str = device
        box_xyxy = None
        box_label = 1
        if use_box and 1 in use_box:
            box_xyxy = (float(x1 or 0), float(y1 or 0), float(x2 or 0), float(y2 or 0))
            box_label = int(bl) if bl is not None else 1
        masks, preview = segment_image_text(
            _sam3_img_bundle,
            _img_sam3,
            text or "",
            box_xyxy=box_xyxy,
            box_label=box_label,
        )
        return html.Div(
            [
                html.Img(src=_pil_to_src(preview), style={"maxWidth": "100%"}),
                html.P(f"Instances: {len(masks)}", className="small"),
            ]
        )

    @app.callback(
        Output("sam3-vid-meta", "data"),
        Input("sam3-upload", "contents"),
        Input("sam3-media", "value"),
        State("sam3-upload", "filename"),
        State("sam3-text", "value"),
    )
    def _sam3_vid_meta(contents: str | None, media: str | None, filename: str | None, text: str | None) -> Any:
        global _raw_sam3_vid, _img_sam3, _frames_sam3, _sam3_vid_result, _sam3_vid_result_key
        if media != "video" or not contents:
            _raw_sam3_vid = None
            _frames_sam3 = None
            return None
        raw = _decode_upload_bytes(contents)
        if raw is None:
            _raw_sam3_vid = None
            return None
        _raw_sam3_vid = raw
        _img_sam3 = None
        _frames_sam3 = None
        _sam3_vid_result = None
        _sam3_vid_result_key = None
        suf = Path(filename or "clip.mp4").suffix or ".mp4"
        return {"suffix": suf, "filename": filename or "", "text": (text or "").strip()}

    @app.callback(
        Output("sam3-vid-body", "children"),
        Input("sam3-vid-meta", "data"),
        Input("sam3-maxf", "value"),
    )
    def _sam3_vid_body(meta: dict[str, Any] | None, max_f: float | None) -> Any:
        global _raw_sam3_vid, _frames_sam3
        if not meta or _raw_sam3_vid is None:
            return html.P("Upload a video.", className="text-muted")
        mf = int(max_f) if max_f is not None else 50
        suf = meta.get("suffix") or ".mp4"
        _frames_sam3 = load_video_frames_from_bytes(_raw_sam3_vid, suffix=suf, max_frames=mf, stride=1)
        return html.Div(
            [
                dcc.Markdown(f"Loaded **{len(_frames_sam3)}** frame(s)."),
                dbc.Button("Load SAM 3 (video) model", id="sam3-vid-load", color="primary", className="me-2"),
                html.Span(id="sam3-vid-load-msg", className="text-success small"),
                dbc.Button("Run SAM 3 (video)", id="sam3-vid-run", color="success", className="mt-2 d-block"),
            ]
        )

    @app.callback(
        Output("sam3-vid-load-msg", "children"),
        Input("sam3-vid-load", "n_clicks"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam3_vid_load(n: int | None, device: str | None) -> str:
        global _sam3_vid_bundle, _device_str
        if not n:
            raise PreventUpdate
        import torch

        if device:
            _device_str = device
        _sam3_vid_bundle = load_sam3_video(device=torch.device(_device_str))
        return "SAM 3 video model ready."

    @app.callback(
        Output("sam3-vid-track-out", "children"),
        Input("sam3-vid-run", "n_clicks"),
        State("sam3-vid-meta", "data"),
        State("sam3-maxf", "value"),
        State("sam3-text", "value"),
        State("device-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _sam3_vid_run(
        n: int | None,
        meta: dict[str, Any] | None,
        max_f: float | None,
        text: str | None,
        device: str | None,
    ) -> Any:
        global _sam3_vid_bundle, _frames_sam3, _sam3_vid_result, _sam3_vid_result_key, _device_str
        if not n or _sam3_vid_bundle is None or _frames_sam3 is None or not meta:
            raise PreventUpdate
        if device:
            _device_str = device
        mf = int(max_f) if max_f is not None else 50
        t = (text or "").strip()
        vid_sig = (meta.get("filename", ""), meta.get("suffix", ""))
        key = (vid_sig, mf, t)
        out, first_preview = track_video_text(_sam3_vid_bundle, _frames_sam3, t, max_frames=mf)
        _sam3_vid_result = (out, first_preview)
        _sam3_vid_result_key = key
        children: list[Any] = [
            html.H6("First frame preview"),
            html.Img(src=_pil_to_src(first_preview), style={"maxWidth": "100%"}),
        ]
        if out:
            sample = {k: _jsonable(v) for k, v in list(out.items())[: min(5, len(out))]}
            children.append(html.Pre(json.dumps(sample, indent=2), className="small bg-light p-2"))
            mxk = max(out.keys())
            children.extend(
                [
                    html.Label("Inspect frame"),
                    dcc.Slider(id="sam3-fi", min=0, max=mxk, step=1, value=0),
                    html.Div(id="sam3-fi-out"),
                ]
            )
        else:
            children.append(dbc.Alert("No frames returned.", color="warning"))
        return html.Div(children)

    @app.callback(
        Output("sam3-fi-out", "children"),
        Input("sam3-fi", "value"),
        State("sam3-vid-meta", "data"),
        State("sam3-maxf", "value"),
        State("sam3-text", "value"),
    )
    def _sam3_vid_frame_inspect(
        fi: float | None,
        meta: dict[str, Any] | None,
        max_f: float | None,
        text: str | None,
    ) -> Any:
        global _sam3_vid_result, _sam3_vid_result_key
        if _sam3_vid_result is None or not meta:
            return html.Div()
        mf = int(max_f) if max_f is not None else 50
        t = (text or "").strip()
        vid_sig = (meta.get("filename", ""), meta.get("suffix", ""))
        key = (vid_sig, mf, t)
        if _sam3_vid_result_key != key:
            return html.Div()
        out, _fp = _sam3_vid_result
        if not out:
            return html.Div()
        i = int(fi or 0)
        if i not in out:
            return html.Div()
        return html.P(f"Frame {i} keys: {list(out[i].keys())}", className="small")


app = create_app()
server = app.server


def main() -> None:
    """
    Run the Dash development server with debug tools and hot reload.

    Editing Python files under this project triggers a server restart so you
    can refresh the browser to see changes.
    """
    app.run(
        debug=True,
        port=8050,
        use_reloader=True,
        dev_tools_hot_reload=True,
    )


if __name__ == "__main__":
    main()
