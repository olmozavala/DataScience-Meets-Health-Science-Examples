import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# Configuration
# =============================================================================

MAX_HISTORY = 200
NUM_SAMPLES = 100
NUM_HIDDEN = 10          # 10 hidden layers
HIDDEN_WIDTH = 32        # every hidden layer has 32 neurons
INIT_SEED = 42

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

COLORS = {
    "bg": "#111111",
    "sidebar": "#1a1a1a",
    "text": "#E1E1E1",
    "accent": "#00ADB5",
    "no_bn": "#FF5722",
    "with_bn": "#2196F3",
}

FIG_H = 290
CARD_CLS = "shadow border-0 mb-3"

# =============================================================================
# Model Definition — 10 hidden layers
# =============================================================================


class DeepNet(nn.Module):
    """x -> [Linear -> act -> [BN]] x 10 -> Linear(W,1)."""

    def __init__(self, use_batchnorm: bool = False, activation: str = "relu") -> None:
        super().__init__()
        act_fn = nn.Tanh if activation == "tanh" else nn.ReLU

        layers: list[nn.Module] = []
        in_dim = 1
        for i in range(NUM_HIDDEN):
            layers.append(nn.Linear(in_dim, HIDDEN_WIDTH))
            layers.append(act_fn())
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(HIDDEN_WIDTH))
            in_dim = HIDDEN_WIDTH

        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(HIDDEN_WIDTH, 1)
        self.use_batchnorm = use_batchnorm

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (prediction, h1_activations, h10_activations).

        h1 = output after hidden block 1 (Linear+act+[BN])
        h10 = output after hidden block 10 (last hidden block)
        """
        # Each hidden block has 2 modules (Linear, act) or 3 (Linear, act, BN)
        block_size = 3 if self.use_batchnorm else 2

        h = x
        h1 = None
        for i in range(NUM_HIDDEN):
            start = i * block_size
            end = start + block_size
            h = self.hidden[start:end](h)
            if i == 0:
                h1 = h.detach()

        h10 = h.detach()
        return self.head(h), h1, h10


# =============================================================================
# Model Serialization
# =============================================================================


def serialize_model(model: nn.Module) -> dict:
    """Convert model state_dict to JSON-serializable dict."""
    return {k: v.cpu().tolist() for k, v in model.state_dict().items()}


def deserialize_model(model: nn.Module, params: dict) -> None:
    """Load JSON-serialized parameters into model, preserving original dtypes."""
    ref = model.state_dict()
    model.load_state_dict(
        {k: torch.tensor(v, dtype=ref[k].dtype) for k, v in params.items()}
    )


def create_model_pair(activation: str = "relu") -> tuple[DeepNet, DeepNet]:
    """Create no-BN and with-BN models that share identical initial linear weights."""
    torch.manual_seed(INIT_SEED)
    no_bn = DeepNet(use_batchnorm=False, activation=activation)
    torch.manual_seed(INIT_SEED)
    with_bn = DeepNet(use_batchnorm=True, activation=activation)
    # Copy all Linear weights/biases from no_bn to with_bn
    with torch.no_grad():
        src = dict(no_bn.named_parameters())
        dst = dict(with_bn.named_parameters())
        for k in src:
            if k in dst and src[k].shape == dst[k].shape:
                dst[k].copy_(src[k])
    return no_bn, with_bn


# =============================================================================
# Data Generation & Statistics
# =============================================================================


def generate_data(
    amplitude: float, offset: float, noise: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate y = amplitude * sin(x) + offset + N(0, noise).

    Returns (x_raw, x_norm, y) where x_norm is standardised to N(0,1).
    """
    np.random.seed(42)
    x = np.linspace(-3, 3, NUM_SAMPLES).astype(np.float32)
    x_norm = ((x - x.mean()) / x.std()).astype(np.float32)
    y = (
        amplitude * np.sin(x)
        + offset
        + np.random.normal(0, noise, NUM_SAMPLES)
    ).astype(np.float32)
    return x, x_norm, y


def compute_ics(
    prev_mean: float, prev_var: float, curr_mean: float, curr_var: float
) -> float:
    """Wasserstein-2 distance between consecutive Gaussian fits of activation distributions."""
    return float(
        np.sqrt(
            (curr_mean - prev_mean) ** 2
            + (np.sqrt(max(curr_var, 1e-12)) - np.sqrt(max(prev_var, 1e-12))) ** 2
        )
    )


def empty_history() -> dict:
    """Fresh history container for one model."""
    return dict(
        params={},
        loss=[],
        h1_mean=[], h1_var=[],
        h10_mean=[], h10_var=[],
        ics_h1=[], ics_h10=[],
        h1_latest=[], h10_latest=[],
        y_pred=[],
    )


def record_snapshot(
    model: nn.Module,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    criterion: nn.Module,
    hist: dict,
) -> dict:
    """Train-mode forward on full dataset: append scalar stats, overwrite latest activations.

    We intentionally keep the model in train() mode so that BatchNorm uses the
    real batch statistics of the full dataset rather than its running estimates
    (which are poorly initialised in the early steps and cause apparent
    oscillation / high ICS that is an artefact, not a real signal).
    """
    model.train()   # <-- keep train mode so BN uses actual batch stats
    with torch.no_grad():
        y_pred, h1, h10 = model(x_all)
        loss = criterion(y_pred, y_all).item()

    h1_np, h10_np = h1.numpy(), h10.numpy()
    h1_m, h1_v = float(h1_np.mean()), float(h1_np.var())
    h10_m, h10_v = float(h10_np.mean()), float(h10_np.var())

    ics1 = (
        compute_ics(hist["h1_mean"][-1], hist["h1_var"][-1], h1_m, h1_v)
        if hist["h1_mean"]
        else 0.0
    )
    ics10 = (
        compute_ics(hist["h10_mean"][-1], hist["h10_var"][-1], h10_m, h10_v)
        if hist["h10_mean"]
        else 0.0
    )

    hist["loss"].append(loss)
    hist["h1_mean"].append(h1_m)
    hist["h1_var"].append(h1_v)
    hist["h10_mean"].append(h10_m)
    hist["h10_var"].append(h10_v)
    hist["ics_h1"].append(ics1)
    hist["ics_h10"].append(ics10)
    hist["h1_latest"] = h1_np.flatten().tolist()
    hist["h10_latest"] = h10_np.flatten().tolist()
    hist["y_pred"] = y_pred.numpy().flatten().tolist()

    for key in ("loss", "h1_mean", "h1_var", "h10_mean", "h10_var", "ics_h1", "ics_h10"):
        if len(hist[key]) > MAX_HISTORY:
            hist[key] = hist[key][-MAX_HISTORY:]

    return hist


# =============================================================================
# Layout
# =============================================================================

sidebar = html.Div(
    [
        html.H2("Config", className="display-6", style={"color": COLORS["accent"]}),
        html.Hr(style={"backgroundColor": COLORS["accent"]}),
        # -- Data --
        html.H5("Data"),
        html.Div(
            [dbc.Label("Amplitude"), dbc.Input(id="amplitude", type="number", value=2.0, step=0.1)],
            className="mb-2",
        ),
        html.Div(
            [dbc.Label("Offset"), dbc.Input(id="offset", type="number", value=0.0, step=0.1)],
            className="mb-2",
        ),
        html.Div(
            [
                dbc.Label("Noise"),
                dcc.Slider(
                    id="noise", min=0, max=3, step=0.1, value=0.3,
                    marks={i: str(i) for i in range(4)},
                ),
            ],
            className="mb-3",
        ),
        # -- Model --
        html.H5("Model", className="mt-2"),
        html.Div(
            [
                dbc.Label("Activation"),
                dbc.Select(
                    id="activation",
                    options=[
                        {"label": "ReLU", "value": "relu"},
                        {"label": "Tanh", "value": "tanh"},
                    ],
                    value="tanh",
                ),
            ],
            className="mb-3",
        ),
        # -- Training --
        html.H5("Training", className="mt-2"),
        html.Div(
            [dbc.Label("Learning Rate"), dbc.Input(id="lr", type="number", value=0.001, step=0.0005)],
            className="mb-2",
        ),
        html.Div(
            [
                dbc.Label("Optimizer"),
                dbc.Select(
                    id="optimizer",
                    options=[
                        {"label": "Adam  (recommended with BN)", "value": "adam"},
                        {"label": "SGD", "value": "sgd"},
                    ],
                    value="adam",
                ),
            ],
            className="mb-2",
        ),
        html.Div(
            [
                dbc.Label("Batch Size"),
                dbc.Input(id="batch-size", type="number", value=20, min=4, max=NUM_SAMPLES, step=1),
            ],
            className="mb-2",
        ),
        html.Div(
            [dbc.Label("Epochs (Run All)"), dbc.Input(id="epochs", type="number", value=50, step=5)],
            className="mb-3",
        ),
        # -- Inspect --
        html.H5("Inspect Layer", className="mt-2"),
        dbc.RadioItems(
            id="layer-select",
            options=[
                {"label": " h(1)  [first hidden]", "value": "h1"},
                {"label": " h(10) [last hidden]", "value": "h10"},
            ],
            value="h1",
            inline=True,
            className="mb-3",
        ),
        html.Hr(),
        # -- Buttons --
        dbc.ButtonGroup(
            [
                dbc.Button("Step", id="step-btn", color="info", outline=True),
                dbc.Button("10 Steps", id="ten-step-btn", color="info", outline=True),
                dbc.Button("All Epochs", id="run-btn", color="primary"),
            ],
            className="w-100 shadow mb-2",
        ),
        dbc.Button(
            "Reset", id="reset-btn", color="danger", outline=True,
            size="sm", className="w-100 mt-1",
        ),
        html.Hr(),
        html.Div(id="stats-output", className="small"),
        dcc.Store(id="model-state"),
    ],
    style=dict(
        padding="1.5rem 1rem",
        backgroundColor=COLORS["sidebar"],
        height="100vh",
        overflowY="auto",
        boxShadow="2px 0 10px rgba(0,0,0,.5)",
    ),
)

content = dbc.Container(
    [
        dbc.Row(
            dbc.Col([
                html.H2(
                    "Internal Covariate Shift & Batch Normalization",
                    className="text-center mt-3 fw-bold",
                ),
                html.P(
                    "Side-by-side comparison: how BatchNorm stabilises hidden-layer distributions "
                    "(10 hidden layers, 32 neurons each)",
                    className="text-center text-muted lead mb-3",
                ),
            ])
        ),
        # Row 1: Loss & Data
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Loss Comparison", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="loss-plot", style={"height": f"{FIG_H}px"})),
                ], className=CARD_CLS),
                lg=7,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Data & Predictions", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="data-plot", style={"height": f"{FIG_H}px"})),
                ], className=CARD_CLS),
                lg=5,
            ),
        ]),
        # Row 2: Activation histograms for h(1) and h(10)
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("h(1) Activation Distribution  [first hidden]", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="h1-hist", style={"height": f"{FIG_H}px"})),
                ], className=CARD_CLS),
                lg=6,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("h(10) Activation Distribution  [last hidden]", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="h10-hist", style={"height": f"{FIG_H}px"})),
                ], className=CARD_CLS),
                lg=6,
            ),
        ]),
        # Row 3: Activation stats & ICS
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Activation Mean & Std Dev", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="stats-plot", style={"height": f"{FIG_H}px"})),
                ], className=CARD_CLS),
                lg=6,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("ICS Metric (Wasserstein-2)", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="ics-plot", style={"height": f"{FIG_H}px"})),
                ], className=CARD_CLS),
                lg=6,
            ),
        ]),
    ],
    fluid=True,
    style={"padding": "0.5rem"},
)

app.layout = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(sidebar, md=3, lg=3, className="p-0"),
                dbc.Col(content, md=9, lg=9, style={"overflowY": "auto", "height": "100vh"}),
            ],
            className="g-0",
        )
    ],
    style={"backgroundColor": COLORS["bg"], "color": COLORS["text"], "overflow": "hidden"},
)


# =============================================================================
# Callback: state management (training / reset)
# =============================================================================


@app.callback(
    Output("model-state", "data"),
    [
        Input("run-btn", "n_clicks"),
        Input("step-btn", "n_clicks"),
        Input("ten-step-btn", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        Input("amplitude", "value"),
        Input("offset", "value"),
        Input("noise", "value"),
        Input("activation", "value"),
    ],
    [
        State("model-state", "data"),
        State("lr", "value"),
        State("epochs", "value"),
        State("batch-size", "value"),
        State("optimizer", "value"),
    ],
)
def manage_state(
    _run: int | None,
    _step: int | None,
    _ten: int | None,
    _reset: int | None,
    amp: float | None,
    off: float | None,
    noise: float | None,
    activation: str | None,
    state: dict | None,
    lr: float | None,
    epochs: int | None,
    batch_size: int | None,
    optimizer_name: str | None,
) -> dict:
    """Create / reset / train both models and return updated JSON state."""
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "init"

    amp = amp if amp is not None else 2.0
    off = off if off is not None else 0.0
    noise = noise if noise is not None else 0.3
    activation = activation or "tanh"
    lr = lr if lr and lr > 0 else 0.001
    batch_size = max(int(batch_size or 20), 4)
    epochs = int(epochs or 50)
    n_batches = (NUM_SAMPLES + batch_size - 1) // batch_size

    # ---- Reset? ----
    reset_triggers = {"reset-btn", "amplitude", "offset", "noise", "activation", "init"}
    if trigger in reset_triggers or state is None:
        x_raw, x_norm, y_np = generate_data(amp, off, noise)
        xt = torch.from_numpy(x_norm).reshape(-1, 1)
        yt = torch.from_numpy(y_np).reshape(-1, 1)
        m0, m1 = create_model_pair(activation)
        crit = nn.MSELoss()
        h0 = record_snapshot(m0, xt, yt, crit, empty_history())
        h1 = record_snapshot(m1, xt, yt, crit, empty_history())
        h0["params"] = serialize_model(m0)
        h1["params"] = serialize_model(m1)
        return dict(step_count=0, activation=activation, no_bn=h0, with_bn=h1)

    # ---- How many steps? ----
    if trigger == "step-btn":
        steps = 1
    elif trigger == "ten-step-btn":
        steps = 10
    elif trigger == "run-btn":
        steps = epochs * n_batches
    else:
        return state

    # ---- Train both models on identical batches ----
    x_raw, x_norm, y_np = generate_data(amp, off, noise)
    xt = torch.from_numpy(x_norm).reshape(-1, 1)
    yt = torch.from_numpy(y_np).reshape(-1, 1)

    act = state.get("activation", "tanh")
    m0 = DeepNet(False, act)
    m1 = DeepNet(True, act)
    deserialize_model(m0, state["no_bn"]["params"])
    deserialize_model(m1, state["with_bn"]["params"])

    optimizer_name = (optimizer_name or "adam").lower()
    if optimizer_name == "adam":
        opt0 = optim.Adam(m0.parameters(), lr=lr)
        opt1 = optim.Adam(m1.parameters(), lr=lr)
    else:
        opt0 = optim.SGD(m0.parameters(), lr=lr)
        opt1 = optim.SGD(m1.parameters(), lr=lr)
    crit = nn.MSELoss()

    h0, h1_hist = state["no_bn"], state["with_bn"]
    base = state["step_count"]

    for i in range(steps):
        idx = (base + i) % n_batches
        s = idx * batch_size
        e = min(s + batch_size, NUM_SAMPLES)
        xb, yb = xt[s:e], yt[s:e]

        # BatchNorm1d requires >= 2 samples; skip a trailing singleton batch
        if xb.shape[0] < 2:
            continue

        for mdl, opt in ((m0, opt0), (m1, opt1)):
            mdl.train()
            opt.zero_grad()
            pred, _, _ = mdl(xb)
            crit(pred, yb).backward()
            opt.step()

        h0 = record_snapshot(m0, xt, yt, crit, h0)
        h1_hist = record_snapshot(m1, xt, yt, crit, h1_hist)

    h0["params"] = serialize_model(m0)
    h1_hist["params"] = serialize_model(m1)
    return dict(step_count=base + steps, activation=act, no_bn=h0, with_bn=h1_hist)


# =============================================================================
# Callback: visualization
# =============================================================================


def _base_layout(**extra: object) -> dict:
    """Common Plotly layout kwargs for all figures."""
    base = dict(
        template="plotly_dark",
        height=FIG_H,
        margin=dict(l=45, r=15, b=40, t=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0.5, xanchor="center"),
    )
    base.update(extra)
    return base


def _make_hist(nb_vals: list, bn_vals: list) -> go.Figure:
    """Overlaid histograms for No-BN vs With-BN activation values."""
    fig = go.Figure(
        layout=_base_layout(barmode="overlay", xaxis_title="Activation value", yaxis_title="Count")
    )
    if nb_vals:
        fig.add_trace(go.Histogram(
            x=nb_vals, nbinsx=50, name="No BN",
            marker_color=COLORS["no_bn"], opacity=0.55,
        ))
    if bn_vals:
        fig.add_trace(go.Histogram(
            x=bn_vals, nbinsx=50, name="With BN",
            marker_color=COLORS["with_bn"], opacity=0.35,
        ))
    return fig


@app.callback(
    [
        Output("loss-plot", "figure"),
        Output("data-plot", "figure"),
        Output("h1-hist", "figure"),
        Output("h10-hist", "figure"),
        Output("stats-plot", "figure"),
        Output("ics-plot", "figure"),
        Output("stats-output", "children"),
    ],
    [Input("model-state", "data"), Input("layer-select", "value")],
    [State("amplitude", "value"), State("offset", "value"), State("noise", "value")],
)
def render(
    state: dict | None,
    layer: str,
    amp: float | None,
    off: float | None,
    noise: float | None,
) -> tuple:
    """Build all six figures and stats text from current state."""
    if state is None:
        return (dash.no_update,) * 7

    amp = amp if amp is not None else 2.0
    off = off if off is not None else 0.0
    noise = noise if noise is not None else 0.3

    nb, bn = state["no_bn"], state["with_bn"]
    step_idx = list(range(len(nb["loss"])))

    # -- 1. Loss comparison --
    fig_loss = go.Figure(
        data=[
            go.Scatter(
                x=step_idx, y=nb["loss"], name="No BN",
                line=dict(color=COLORS["no_bn"], width=2),
            ),
            go.Scatter(
                x=step_idx, y=bn["loss"], name="With BN",
                line=dict(color=COLORS["with_bn"], width=2),
            ),
        ],
        layout=_base_layout(xaxis_title="Step", yaxis_title="MSE Loss", yaxis_type="log"),
    )

    # -- 2. Data & predictions --
    x_np, _, y_np = generate_data(amp, off, noise)
    x_fine = np.linspace(-3, 3, 200)

    fig_data = go.Figure(layout=_base_layout(xaxis_title="x", yaxis_title="y"))
    fig_data.add_trace(go.Scatter(
        x=x_np, y=y_np, mode="markers",
        marker=dict(color="white", opacity=0.4, size=4), name="Data",
    ))
    fig_data.add_trace(go.Scatter(
        x=x_fine, y=amp * np.sin(x_fine) + off,
        mode="lines", line=dict(color="cyan", dash="dash", width=1), name="True f(x)",
    ))
    if nb.get("y_pred"):
        fig_data.add_trace(go.Scatter(
            x=x_np, y=nb["y_pred"], mode="lines",
            line=dict(color=COLORS["no_bn"], width=2), name="No BN",
        ))
    if bn.get("y_pred"):
        fig_data.add_trace(go.Scatter(
            x=x_np, y=bn["y_pred"], mode="lines",
            line=dict(color=COLORS["with_bn"], width=2), name="With BN",
        ))

    # -- 3 & 4. Activation histograms for h(1) and h(10) --
    fig_h1 = _make_hist(nb.get("h1_latest", []), bn.get("h1_latest", []))
    fig_h10 = _make_hist(nb.get("h10_latest", []), bn.get("h10_latest", []))
    fig_h10.update_xaxes(range=[-2, 2])
    fig_h10.update_yaxes(range=[0, 600])

    # -- 5. Mean & Std evolution for selected layer --
    mk = f"{layer}_mean"
    vk = f"{layer}_var"
    layer_label = "h(1)" if layer == "h1" else "h(10)"

    fig_stats = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        subplot_titles=(f"{layer_label} Mean", f"{layer_label} Std Dev"),
        vertical_spacing=0.18,
    )
    fig_stats.add_trace(go.Scatter(
        x=step_idx, y=nb[mk], name="No BN",
        line=dict(color=COLORS["no_bn"], width=2),
    ), row=1, col=1)
    fig_stats.add_trace(go.Scatter(
        x=step_idx, y=bn[mk], name="With BN",
        line=dict(color=COLORS["with_bn"], width=2),
    ), row=1, col=1)

    nb_std = [np.sqrt(max(v, 0)) for v in nb[vk]]
    bn_std = [np.sqrt(max(v, 0)) for v in bn[vk]]
    fig_stats.add_trace(go.Scatter(
        x=step_idx, y=nb_std, showlegend=False,
        line=dict(color=COLORS["no_bn"], width=2, dash="dot"),
    ), row=2, col=1)
    fig_stats.add_trace(go.Scatter(
        x=step_idx, y=bn_std, showlegend=False,
        line=dict(color=COLORS["with_bn"], width=2, dash="dot"),
    ), row=2, col=1)
    fig_stats.update_layout(
        template="plotly_dark", height=FIG_H,
        margin=dict(l=45, r=15, b=40, t=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, x=0.5, xanchor="center"),
    )
    fig_stats.update_xaxes(title_text="Step", row=2, col=1)

    # -- 6. ICS metric for selected layer --
    ics_key = f"ics_{layer}"
    fig_ics = go.Figure(
        data=[
            go.Scatter(
                x=step_idx, y=nb[ics_key], name="No BN",
                line=dict(color=COLORS["no_bn"], width=2),
            ),
            go.Scatter(
                x=step_idx, y=bn[ics_key], name="With BN",
                line=dict(color=COLORS["with_bn"], width=2),
            ),
        ],
        layout=_base_layout(xaxis_title="Step", yaxis_title=f"ICS  W\u2082 ({layer_label})"),
    )

    # -- Stats text --
    info = [
        html.P([html.B("Steps: "), f"{state['step_count']}"]),
        html.P(
            [html.B("No BN loss: "), f"{nb['loss'][-1]:.6f}"] if nb["loss"] else "",
            style={"color": COLORS["no_bn"]},
        ),
        html.P(
            [html.B("BN loss: "), f"{bn['loss'][-1]:.6f}"] if bn["loss"] else "",
            style={"color": COLORS["with_bn"]},
        ),
        html.Hr(),
        html.P(
            [html.B(f"No BN {layer_label} mean: "),
             f"{nb[mk][-1]:.4f}"] if nb[mk] else "",
            style={"color": COLORS["no_bn"]},
        ),
        html.P(
            [html.B(f"BN {layer_label} mean: "),
             f"{bn[mk][-1]:.4f}"] if bn[mk] else "",
            style={"color": COLORS["with_bn"]},
        ),
        html.P(
            [html.B(f"No BN {layer_label} std: "),
             f"{np.sqrt(max(nb[vk][-1], 0)):.4f}"] if nb[vk] else "",
            style={"color": COLORS["no_bn"]},
        ),
        html.P(
            [html.B(f"BN {layer_label} std: "),
             f"{np.sqrt(max(bn[vk][-1], 0)):.4f}"] if bn[vk] else "",
            style={"color": COLORS["with_bn"]},
        ),
    ]

    return fig_loss, fig_data, fig_h1, fig_h10, fig_stats, fig_ics, info


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8050)
