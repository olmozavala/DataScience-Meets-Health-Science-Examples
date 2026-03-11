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
NUM_HIDDEN = 10
HIDDEN_WIDTH = 32
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
    "no_reg": "#FF5722",
    "l2_reg": "#2196F3",
    "l1_reg": "#4CAF50",
}

FIG_H = 290
CARD_CLS = "shadow border-0 mb-3"

# =============================================================================
# Model Definition
# =============================================================================


class DeepNet(nn.Module):
    """x -> [Linear -> act] x NUM_HIDDEN -> Linear(W, 1)."""

    def __init__(self, activation: str = "relu") -> None:
        super().__init__()
        act_fn = nn.Tanh if activation == "tanh" else nn.ReLU

        layers: list[nn.Module] = []
        in_dim = 1
        for _ in range(NUM_HIDDEN):
            layers.append(nn.Linear(in_dim, HIDDEN_WIDTH))
            layers.append(act_fn())
            in_dim = HIDDEN_WIDTH

        self.hidden = nn.Sequential(*layers)
        self.head = nn.Linear(HIDDEN_WIDTH, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning predictions only."""
        return self.head(self.hidden(x))


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


def create_model_triplet(activation: str = "relu") -> tuple[DeepNet, DeepNet, DeepNet]:
    """Create 3 models with identical initial weights for no-reg, L2, and L1."""
    torch.manual_seed(INIT_SEED)
    m_none = DeepNet(activation=activation)
    torch.manual_seed(INIT_SEED)
    m_l2 = DeepNet(activation=activation)
    torch.manual_seed(INIT_SEED)
    m_l1 = DeepNet(activation=activation)
    return m_none, m_l2, m_l1


# =============================================================================
# Data Generation & Weight Statistics
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


def get_layer_weights(model: nn.Module, layer_idx: int) -> np.ndarray:
    """Extract flattened weight tensor from a specific hidden linear layer.

    Parameters
    ----------
    model : nn.Module
        The DeepNet model.
    layer_idx : int
        Index of the hidden block (0-based). Each block has 2 modules (Linear, act).

    Returns
    -------
    np.ndarray
        Flattened weight values from the Linear layer in the specified block.
    """
    linear_module = model.hidden[layer_idx * 2]
    return linear_module.weight.detach().cpu().numpy().flatten()


def get_all_weights(model: nn.Module) -> np.ndarray:
    """Extract all weight parameters (no biases) concatenated into a single array."""
    all_w: list[np.ndarray] = []
    for name, param in model.named_parameters():
        if "weight" in name:
            all_w.append(param.detach().cpu().numpy().flatten())
    return np.concatenate(all_w)


def pct_near_zero(weights: np.ndarray, threshold: float) -> float:
    """Percentage of weights whose absolute value is below threshold."""
    return float(np.mean(np.abs(weights) < threshold) * 100)


def l1_penalty(model: nn.Module) -> torch.Tensor:
    """Compute the L1 norm of all weight parameters."""
    penalty = torch.tensor(0.0)
    for name, param in model.named_parameters():
        if "weight" in name:
            penalty = penalty + param.abs().sum()
    return penalty


def empty_history() -> dict:
    """Fresh history container for one model."""
    return dict(
        params={},
        loss=[],
        pct_zero_all=[],
        pct_zero_h1=[],
        pct_zero_h10=[],
        weight_norm=[],
        h1_weights=[],
        h10_weights=[],
        all_weights=[],
        y_pred=[],
    )


def record_snapshot(
    model: nn.Module,
    x_all: torch.Tensor,
    y_all: torch.Tensor,
    criterion: nn.Module,
    hist: dict,
    threshold: float,
) -> dict:
    """Evaluate model and record weight statistics into history dict.

    Parameters
    ----------
    model : nn.Module
        The network to evaluate.
    x_all : torch.Tensor
        Full input tensor.
    y_all : torch.Tensor
        Full target tensor.
    criterion : nn.Module
        Loss function (MSE).
    hist : dict
        History dict to append to.
    threshold : float
        Threshold for near-zero weight counting.

    Returns
    -------
    dict
        Updated history dict.
    """
    model.eval()
    with torch.no_grad():
        y_pred = model(x_all)
        loss = criterion(y_pred, y_all).item()

    h1_w = get_layer_weights(model, 0)
    h10_w = get_layer_weights(model, NUM_HIDDEN - 1)
    all_w = get_all_weights(model)

    hist["loss"].append(loss)
    hist["pct_zero_all"].append(pct_near_zero(all_w, threshold))
    hist["pct_zero_h1"].append(pct_near_zero(h1_w, threshold))
    hist["pct_zero_h10"].append(pct_near_zero(h10_w, threshold))
    hist["weight_norm"].append(float(np.linalg.norm(all_w)))

    hist["h1_weights"] = h1_w.tolist()
    hist["h10_weights"] = h10_w.tolist()
    hist["all_weights"] = all_w.tolist()
    hist["y_pred"] = y_pred.detach().numpy().flatten().tolist()

    for key in ("loss", "pct_zero_all", "pct_zero_h1", "pct_zero_h10", "weight_norm"):
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
                dbc.Label("Batch Size"),
                dbc.Input(id="batch-size", type="number", value=20, min=4, max=NUM_SAMPLES, step=1),
            ],
            className="mb-2",
        ),
        html.Div(
            [dbc.Label("Epochs (Run All)"), dbc.Input(id="epochs", type="number", value=50, step=5)],
            className="mb-3",
        ),
        # -- Regularization --
        html.H5("Regularization", className="mt-2"),
        html.Div(
            [
                dbc.Label("Lambda (reg strength)"),
                dbc.Input(id="reg-lambda", type="number", value=0.01, step=0.001, min=0),
            ],
            className="mb-2",
        ),
        html.Div(
            [
                dbc.Label("Near-zero threshold"),
                dbc.Input(id="zero-threshold", type="number", value=0.01, step=0.005, min=0.001),
            ],
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
                    "Effects of Regularization on Weight Distributions",
                    className="text-center mt-3 fw-bold",
                ),
                html.P(
                    "Comparing No Regularization, L2 (Ridge), and L1 (Lasso) on a deep network "
                    f"({NUM_HIDDEN} hidden layers, {HIDDEN_WIDTH} neurons each)",
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
        # Row 2: Weight histograms for h(1) and h(10)
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("h(1) Weight Distribution  [first hidden]", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="h1-hist", style={"height": f"{FIG_H}px"})),
                ], className=CARD_CLS),
                lg=6,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("h(10) Weight Distribution  [last hidden]", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="h10-hist", style={"height": f"{FIG_H}px"})),
                ], className=CARD_CLS),
                lg=6,
            ),
        ]),
        # Row 3: All-weights histogram & near-zero percentage
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("All Weights Distribution", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="all-hist", style={"height": f"{FIG_H}px"})),
                ], className=CARD_CLS),
                lg=6,
            ),
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("% Weights Near Zero (|w| < threshold)", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="pct-zero-plot", style={"height": f"{FIG_H}px"})),
                ], className=CARD_CLS),
                lg=6,
            ),
        ]),
        # Row 4: Weight norm evolution
        dbc.Row([
            dbc.Col(
                dbc.Card([
                    dbc.CardHeader("Total Weight Norm Over Training", className="fw-bold"),
                    dbc.CardBody(dcc.Graph(id="norm-plot", style={"height": f"{FIG_H}px"})),
                ], className=CARD_CLS),
                lg=12,
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
        State("reg-lambda", "value"),
        State("zero-threshold", "value"),
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
    reg_lambda: float | None,
    threshold: float | None,
) -> dict:
    """Create / reset / train all three models and return updated JSON state."""
    ctx = dash.callback_context
    trigger = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else "init"

    amp = amp if amp is not None else 2.0
    off = off if off is not None else 0.0
    noise = noise if noise is not None else 0.3
    activation = activation or "tanh"
    lr = lr if lr and lr > 0 else 0.001
    batch_size = max(int(batch_size or 20), 4)
    epochs = int(epochs or 50)
    reg_lambda = reg_lambda if reg_lambda is not None and reg_lambda >= 0 else 0.01
    threshold = threshold if threshold is not None and threshold > 0 else 0.01
    n_batches = (NUM_SAMPLES + batch_size - 1) // batch_size

    # ---- Reset? ----
    reset_triggers = {"reset-btn", "amplitude", "offset", "noise", "activation", "init"}
    if trigger in reset_triggers or state is None:
        x_raw, x_norm, y_np = generate_data(amp, off, noise)
        xt = torch.from_numpy(x_norm).reshape(-1, 1)
        yt = torch.from_numpy(y_np).reshape(-1, 1)
        m_none, m_l2, m_l1 = create_model_triplet(activation)
        crit = nn.MSELoss()
        h_none = record_snapshot(m_none, xt, yt, crit, empty_history(), threshold)
        h_l2 = record_snapshot(m_l2, xt, yt, crit, empty_history(), threshold)
        h_l1 = record_snapshot(m_l1, xt, yt, crit, empty_history(), threshold)
        h_none["params"] = serialize_model(m_none)
        h_l2["params"] = serialize_model(m_l2)
        h_l1["params"] = serialize_model(m_l1)
        return dict(
            step_count=0, activation=activation,
            no_reg=h_none, l2_reg=h_l2, l1_reg=h_l1,
        )

    # ---- How many steps? ----
    if trigger == "step-btn":
        steps = 1
    elif trigger == "ten-step-btn":
        steps = 10
    elif trigger == "run-btn":
        steps = epochs * n_batches
    else:
        return state

    # ---- Train all three models on identical batches ----
    x_raw, x_norm, y_np = generate_data(amp, off, noise)
    xt = torch.from_numpy(x_norm).reshape(-1, 1)
    yt = torch.from_numpy(y_np).reshape(-1, 1)

    act = state.get("activation", "tanh")
    m_none = DeepNet(act)
    m_l2 = DeepNet(act)
    m_l1 = DeepNet(act)
    deserialize_model(m_none, state["no_reg"]["params"])
    deserialize_model(m_l2, state["l2_reg"]["params"])
    deserialize_model(m_l1, state["l1_reg"]["params"])

    opt_none = optim.Adam(m_none.parameters(), lr=lr)
    opt_l2 = optim.Adam(m_l2.parameters(), lr=lr, weight_decay=reg_lambda)
    opt_l1 = optim.Adam(m_l1.parameters(), lr=lr)
    crit = nn.MSELoss()

    h_none = state["no_reg"]
    h_l2 = state["l2_reg"]
    h_l1 = state["l1_reg"]
    base = state["step_count"]

    for i in range(steps):
        idx = (base + i) % n_batches
        s = idx * batch_size
        e = min(s + batch_size, NUM_SAMPLES)
        xb, yb = xt[s:e], yt[s:e]

        if xb.shape[0] < 2:
            continue

        # No regularization
        m_none.train()
        opt_none.zero_grad()
        crit(m_none(xb), yb).backward()
        opt_none.step()

        # L2 (weight_decay in Adam is equivalent to L2 regularization)
        m_l2.train()
        opt_l2.zero_grad()
        crit(m_l2(xb), yb).backward()
        opt_l2.step()

        # L1 (manual penalty added to loss)
        m_l1.train()
        opt_l1.zero_grad()
        loss_l1 = crit(m_l1(xb), yb) + reg_lambda * l1_penalty(m_l1)
        loss_l1.backward()
        opt_l1.step()

        h_none = record_snapshot(m_none, xt, yt, crit, h_none, threshold)
        h_l2 = record_snapshot(m_l2, xt, yt, crit, h_l2, threshold)
        h_l1 = record_snapshot(m_l1, xt, yt, crit, h_l1, threshold)

    h_none["params"] = serialize_model(m_none)
    h_l2["params"] = serialize_model(m_l2)
    h_l1["params"] = serialize_model(m_l1)
    return dict(
        step_count=base + steps, activation=act,
        no_reg=h_none, l2_reg=h_l2, l1_reg=h_l1,
    )


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


def _make_weight_hist(
    none_w: list, l2_w: list, l1_w: list, title: str = ""
) -> go.Figure:
    """Overlaid histograms of weight values for the three models."""
    fig = go.Figure(
        layout=_base_layout(
            barmode="overlay", xaxis_title="Weight value", yaxis_title="Count",
        )
    )
    if none_w:
        fig.add_trace(go.Histogram(
            x=none_w, nbinsx=60, name="No Reg",
            marker_color=COLORS["no_reg"], opacity=0.55,
        ))
    if l2_w:
        fig.add_trace(go.Histogram(
            x=l2_w, nbinsx=60, name="L2",
            marker_color=COLORS["l2_reg"], opacity=0.45,
        ))
    if l1_w:
        fig.add_trace(go.Histogram(
            x=l1_w, nbinsx=60, name="L1",
            marker_color=COLORS["l1_reg"], opacity=0.35,
        ))
    return fig


@app.callback(
    [
        Output("loss-plot", "figure"),
        Output("data-plot", "figure"),
        Output("h1-hist", "figure"),
        Output("h10-hist", "figure"),
        Output("all-hist", "figure"),
        Output("pct-zero-plot", "figure"),
        Output("norm-plot", "figure"),
        Output("stats-output", "children"),
    ],
    [Input("model-state", "data")],
    [
        State("amplitude", "value"),
        State("offset", "value"),
        State("noise", "value"),
        State("zero-threshold", "value"),
    ],
)
def render(
    state: dict | None,
    amp: float | None,
    off: float | None,
    noise: float | None,
    threshold: float | None,
) -> tuple:
    """Build all figures and stats text from current state."""
    if state is None:
        return (dash.no_update,) * 8

    amp = amp if amp is not None else 2.0
    off = off if off is not None else 0.0
    noise = noise if noise is not None else 0.3
    threshold = threshold if threshold is not None and threshold > 0 else 0.01

    nr = state["no_reg"]
    l2 = state["l2_reg"]
    l1 = state["l1_reg"]
    step_idx = list(range(len(nr["loss"])))

    trace_cfg = [
        ("No Reg", COLORS["no_reg"], nr),
        ("L2", COLORS["l2_reg"], l2),
        ("L1", COLORS["l1_reg"], l1),
    ]

    # -- 1. Loss comparison --
    fig_loss = go.Figure(
        data=[
            go.Scatter(
                x=step_idx, y=h["loss"], name=name,
                line=dict(color=color, width=2),
            )
            for name, color, h in trace_cfg
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
    for name, color, h in trace_cfg:
        if h.get("y_pred"):
            fig_data.add_trace(go.Scatter(
                x=x_np, y=h["y_pred"], mode="lines",
                line=dict(color=color, width=2), name=name,
            ))

    # -- 3 & 4. Weight histograms for h(1) and h(10) --
    fig_h1 = _make_weight_hist(
        nr.get("h1_weights", []),
        l2.get("h1_weights", []),
        l1.get("h1_weights", []),
    )
    fig_h10 = _make_weight_hist(
        nr.get("h10_weights", []),
        l2.get("h10_weights", []),
        l1.get("h10_weights", []),
    )

    # -- 5. All weights histogram --
    fig_all = _make_weight_hist(
        nr.get("all_weights", []),
        l2.get("all_weights", []),
        l1.get("all_weights", []),
    )

    # -- 6. Percentage of near-zero weights over training --
    fig_pct = go.Figure(layout=_base_layout(
        xaxis_title="Step", yaxis_title=f"% weights |w| < {threshold}",
    ))
    for name, color, h in trace_cfg:
        fig_pct.add_trace(go.Scatter(
            x=step_idx, y=h["pct_zero_all"], name=name,
            line=dict(color=color, width=2),
        ))

    # -- 7. Weight norm evolution --
    fig_norm = go.Figure(layout=_base_layout(xaxis_title="Step", yaxis_title="L2 Norm of all weights"))
    for name, color, h in trace_cfg:
        fig_norm.add_trace(go.Scatter(
            x=step_idx, y=h["weight_norm"], name=name,
            line=dict(color=color, width=2),
        ))

    # -- Stats text --
    info = [
        html.P([html.B("Steps: "), f"{state['step_count']}"]),
        html.Hr(),
    ]
    for name, color, h in trace_cfg:
        loss_str = f"{h['loss'][-1]:.6f}" if h["loss"] else "N/A"
        pct_str = f"{h['pct_zero_all'][-1]:.1f}%" if h["pct_zero_all"] else "N/A"
        norm_str = f"{h['weight_norm'][-1]:.2f}" if h["weight_norm"] else "N/A"
        info.append(
            html.Div([
                html.P([html.B(f"{name} loss: "), loss_str], style={"color": color}, className="mb-0"),
                html.P([html.B(f"  near-zero: "), pct_str], style={"color": color}, className="mb-0"),
                html.P([html.B(f"  weight norm: "), norm_str], style={"color": color}, className="mb-1"),
            ])
        )

    return fig_loss, fig_data, fig_h1, fig_h10, fig_all, fig_pct, fig_norm, info


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    app.run(debug=True, port=8051)
