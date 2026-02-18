import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# Dashboard Configuration & Styling
# =============================================================================

app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

# Premium colors and style
COLORS = {
    'background': '#111111',
    'sidebar': '#1a1a1a',
    'text': '#E1E1E1',
    'accent': '#00ADB5',
    'path': '#FF5722',
    'data': '#393E46',
}

# =============================================================================
# Layout Definition
# =============================================================================

sidebar = html.Div(
    [
        html.H2("Config", className="display-6", style={'color': COLORS['accent']}),
        html.Hr(style={'backgroundColor': COLORS['accent']}),
        
        html.Div([
            dbc.Label("Data Slope (m_true)"),
            dbc.Input(id="true-slope", type="number", value=1.5, step=0.1),
        ], className="mb-3"),
        
        html.Div([
            dbc.Label("Data Intercept (b_true)"),
            dbc.Input(id="true-intercept", type="number", value=2.0, step=0.1),
        ], className="mb-3"),
        
        html.Div([
            dbc.Label("Noise Level"),
            dcc.Slider(
                id="noise-level",
                min=0, max=5, step=0.1, value=0.5,
                marks={i: str(i) for i in range(6)},
            ),
        ], className="mb-4"),
        
        html.H4("Training", className="mt-4"),
        html.Div([
            dbc.Label("Learning Rate"),
            dbc.Input(id="learning-rate", type="number", value=0.01, step=0.001),
        ], className="mb-3"),
        
        html.Div([
            dbc.Label("Epochs (for 'All')"),
            dbc.Input(id="epochs", type="number", value=50, step=5),
        ], className="mb-3"),
        
        dbc.ButtonGroup(
            [
                dbc.Button("Step", id="step-btn", color="info", outline=True),
                dbc.Button("10 Steps", id="ten-step-btn", color="info", outline=True),
                dbc.Button("All Epochs", id="run-btn", color="primary"),
            ],
            vertical=False,
            className="w-100 shadow mb-2",
        ),
        dbc.Button("Reset Model", id="reset-btn", color="danger", outline=True, size="sm", className="w-100 mt-2"),
        
        dcc.Store(id='model-state'), # Stores current w, b, and history
    ],
    style={
        "padding": "2rem 1rem",
        "backgroundColor": COLORS['sidebar'],
        "height": "100vh",
        "boxShadow": "2px 0px 10px rgba(0,0,0,0.5)"
    }
)

content = dbc.Container(
    [
        dbc.Row([
            dbc.Col(
                [
                    html.H1("Single Neuron: Loss Surface & Optimization", className="text-center mt-4 fw-bold"),
                    html.P("Watching Gradient Descent navigate the MSE Loss Paraboloid", className="text-center text-muted lead mb-4"),
                ], width=12
            )
        ]),
        dbc.Row([
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader("3D Loss Surface (MSE)", className="fw-bold"),
                        dbc.CardBody(
                            dcc.Loading(
                                dcc.Graph(id="loss-surface-3d", style={"height": "65vh"}),
                                type="cube", color=COLORS['accent']
                            )
                        )
                    ], className="shadow-lg border-0"
                ), width=12, lg=8
            ),
            dbc.Col(
                [
                    dbc.Card(
                        [
                            dbc.CardHeader("Linear Regression Fit", className="fw-bold"),
                            dbc.CardBody(
                                dcc.Graph(id="data-fit-2d", style={"height": "35vh"})
                            )
                        ], className="shadow mb-4 border-0"
                    ),
                    dbc.Card(
                        [
                            dbc.CardHeader("Training Statistics", className="fw-bold"),
                            dbc.CardBody(
                                html.Div(id="stats-output", className="text-center small")
                            )
                        ], className="shadow border-0"
                    )
                ], width=12, lg=4
            )
        ])
    ],
    fluid=True,
    style={"padding": "1rem"}
)

app.layout = html.Div(
    [
        dbc.Row([
            dbc.Col(sidebar, width=12, md=3, lg=3, className="p-0"),
            dbc.Col(content, width=12, md=9, lg=9, style={"overflowY": "auto", "height": "100vh"})
        ], className="g-0")
    ],
    style={"backgroundColor": COLORS['background'], "color": COLORS['text'], "overflow": "hidden"}
)

# =============================================================================
# Helper Functions
# =============================================================================

def calculate_loss_surface(x, y, w_range, b_range):
    W, B = np.meshgrid(w_range, b_range)
    Z = np.zeros_like(W)
    for i in range(len(w_range)):
        for j in range(len(b_range)):
            y_pred = w_range[i] * x + b_range[j]
            Z[j, i] = np.mean((y - y_pred)**2)
    return W, B, Z

# =============================================================================
# Callbacks
# =============================================================================

@app.callback(
    Output("model-state", "data"),
    [Input("run-btn", "n_clicks"),
     Input("step-btn", "n_clicks"),
     Input("ten-step-btn", "n_clicks"),
     Input("reset-btn", "n_clicks"),
     Input("true-slope", "value"),
     Input("true-intercept", "value"),
     Input("noise-level", "value")],
    [State("model-state", "data"),
     State("learning-rate", "value"),
     State("epochs", "value")]
)
def manage_model_state(n_run, n_step, n_10, n_reset, m_true, b_true, noise, state, lr, epochs):
    ctx = dash.callback_context
    if not ctx.triggered:
        trigger = 'init'
    else:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]

    # Reset detection: if params changed or reset button clicked
    is_reset_trigger = trigger in ["reset-btn", "true-slope", "true-intercept", "noise-level", "init"]
    
    if is_reset_trigger or state is None:
        # Initialize
        w_init = float(m_true - 5.0 if m_true > 0 else m_true + 5.0)
        b_init = float(b_true - 5.0)
        
        # Calculate initial loss
        np.random.seed(42)
        x_np = np.linspace(-5, 5, 40)
        y_np = m_true * x_np + b_true + np.random.normal(0, noise, size=40)
        loss_init = float(np.mean((y_np - (w_init * x_np + b_init))**2))
        
        return {
            'w': w_init,
            'b': b_init,
            'path_w': [w_init],
            'path_b': [b_init],
            'path_loss': [loss_init],
            'step_count': 0
        }

    # Training logic
    steps_to_run = 0
    if trigger == "step-btn": steps_to_run = 1
    elif trigger == "ten-step-btn": steps_to_run = 10
    elif trigger == "run-btn": steps_to_run = epochs

    if steps_to_run > 0:
        # Data
        np.random.seed(42)
        x_np = np.linspace(-5, 5, 40).astype(np.float32)
        y_np = m_true * x_np + b_true + np.random.normal(0, noise, size=40).astype(np.float32)
        x_torch = torch.from_numpy(x_np).reshape(-1, 1)
        y_torch = torch.from_numpy(y_np).reshape(-1, 1)

        # Build Model from state
        class SimpleNeuron(nn.Module):
            def __init__(self, w, b):
                super().__init__()
                self.linear = nn.Linear(1, 1)
                self.linear.weight.data.fill_(w)
                self.linear.bias.data.fill_(b)
            def forward(self, x): return self.linear(x)

        model = SimpleNeuron(state['w'], state['b'])
        optimizer = optim.SGD(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        new_path_w = state['path_w']
        new_path_b = state['path_b']
        new_path_loss = state['path_loss']

        for _ in range(steps_to_run):
            optimizer.zero_grad()
            loss_val = criterion(model(x_torch), y_torch)
            loss_val.backward()
            optimizer.step()
            
            w = model.linear.weight.item()
            b = model.linear.bias.item()
            l = criterion(model(x_torch), y_torch).item()
            
            new_path_w.append(float(w))
            new_path_b.append(float(b))
            new_path_loss.append(float(l))

        return {
            'w': float(w),
            'b': float(b),
            'path_w': new_path_w,
            'path_b': new_path_b,
            'path_loss': new_path_loss,
            'step_count': state['step_count'] + steps_to_run
        }
    
    return state

@app.callback(
    [Output("loss-surface-3d", "figure"),
     Output("data-fit-2d", "figure"),
     Output("stats-output", "children")],
    [Input("model-state", "data")],
    [State("true-slope", "value"),
     State("true-intercept", "value"),
     State("noise-level", "value")]
)
def update_visuals(state, m_true, b_true, noise):
    if state is None: return dash.no_update, dash.no_update, ""

    # 1. Data Generation
    np.random.seed(42)
    x_np = np.linspace(-5, 5, 40).astype(np.float32)
    y_np = m_true * x_np + b_true + np.random.normal(0, noise, size=40).astype(np.float32)

    # 2. Surface Calculation (Larger Domain)
    # Range is ±8 from true value to see the shape well
    w_grid = np.linspace(m_true - 8, m_true + 8, 50)
    b_grid = np.linspace(b_true - 8, b_true + 8, 50)
    W, B, Z = calculate_loss_surface(x_np, y_np, w_grid, b_grid)

    # 3. Visualization - 3D Surface
    fig_3d = go.Figure()
    fig_3d.add_trace(go.Surface(
        x=w_grid, y=b_grid, z=Z,
        colorscale='Viridis', opacity=0.8,
        showscale=True, name="Loss Surface",
        colorbar=dict(title="MSE", thickness=15, len=0.5)
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=state['path_w'], y=state['path_b'], z=state['path_loss'],
        mode='lines+markers',
        line=dict(color=COLORS['path'], width=5),
        marker=dict(size=4, color=COLORS['path']),
        name="GD Path"
    ))
    # Mark True Solution
    true_loss = np.mean((y_np - (m_true * x_np + b_true))**2)
    fig_3d.add_trace(go.Scatter3d(
        x=[m_true], y=[b_true], z=[true_loss],
        mode='markers', marker=dict(size=10, color='cyan', symbol='diamond'),
        name="Global Minimum"
    ))

    fig_3d.update_layout(
        template="plotly_dark",
        scene=dict(
            xaxis_title="Weight (w)", yaxis_title="Bias (b)", zaxis_title="Loss",
            # Set fixed range to keep the "U" shape visible even when zoomed in
            xaxis=dict(range=[m_true - 8, m_true + 8]),
            yaxis=dict(range=[b_true - 8, b_true + 8]),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.4))
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # 4. Visualization - 2D Fit
    fig_2d = go.Figure()
    fig_2d.add_trace(go.Scatter(
        x=x_np, y=y_np, mode='markers',
        marker=dict(color='white', opacity=0.5), name="Data Points"
    ))
    # True Line
    fig_2d.add_trace(go.Scatter(
        x=x_np, y=m_true * x_np + b_true, mode='lines',
        line=dict(color='cyan', dash='dash'), name="Goal Line"
    ))
    # Current Learned Line
    fig_2d.add_trace(go.Scatter(
        x=x_np, y=state['w'] * x_np + state['b'], mode='lines',
        line=dict(color=COLORS['path'], width=3), name="Current Model"
    ))
    fig_2d.update_layout(
        template="plotly_dark", height=300, 
        margin=dict(l=40, r=20, b=40, t=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    stats = [
        html.P([html.B("Target: "), f"w={m_true:.2f}, b={b_true:.2f}"]),
        html.P([html.B("Current: "), f"w={state['w']:.4f}, b={state['b']:.4f}"]),
        html.P([html.B("Steps: "), f"{state['step_count']}"]),
        html.P([html.B("Loss: "), f"{state['path_loss'][-1]:.6f}"], className="text-info font-monospace")
    ]

    return fig_3d, fig_2d, stats

if __name__ == "__main__":
    app.run(debug=True, port=8050)

