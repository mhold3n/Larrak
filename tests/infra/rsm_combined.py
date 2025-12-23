"""
Generate Combined RSM Visualization.
Displays all 15 Phase 1 3D response surfaces (5 phi levels × 3 metrics) on a single
interactive Plotly chart with dropdown checkbox toggles for filtering.
"""

import os
import sys
from typing import Any

# Add project root to sys.path for direct execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata

from tests.infra.sensitivity_dashboard import load_matrix_data


# Color palettes for each metric (5 colors per metric, one per phi level)
METRIC_COLORS = {
    "gas_kinetic_efficiency": [
        "Viridis",  # phi=0.8
        "Cividis",  # phi=0.9
        "Plasma",  # phi=1.0
        "Inferno",  # phi=1.1
        "Magma",  # phi=1.2
    ],
    "thermal_efficiency": [
        "Blues",
        "Purples",
        "Greens",
        "Oranges",
        "Reds",
    ],
    "peak_pressure_bar": [
        "YlGn",
        "YlOrBr",
        "BuPu",
        "PuRd",
        "GnBu",
    ],
}

PHI_LEVELS = {
    "Lean (phi=0.8)": 0.8,
    "Slightly Lean (phi=0.9)": 0.9,
    "Stoic (phi=1.0)": 1.0,
    "Slightly Rich (phi=1.1)": 1.1,
    "Rich (phi=1.2)": 1.2,
}

METRICS = [
    ("gas_kinetic_efficiency", "Gas-Kinetic Efficiency"),
    ("thermal_efficiency", "Thermal Efficiency"),
    ("peak_pressure_bar", "Peak Pressure (bar)"),
]


def interpolate_surface(
    df: pd.DataFrame, x_col: str, y_col: str, z_col: str, resolution: int = 25
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Interpolate a 2D surface from scattered data."""
    df_clean = df.dropna(subset=[x_col, y_col, z_col])
    df_clean = df_clean[~df_clean[[x_col, y_col, z_col]].isin([np.inf, -np.inf]).any(axis=1)]

    if len(df_clean) < 4:
        return None, None, None

    x_min, x_max = df_clean[x_col].min(), df_clean[x_col].max()
    y_min, y_max = df_clean[y_col].min(), df_clean[y_col].max()

    x_lin = np.linspace(x_min, x_max, resolution)
    y_lin = np.linspace(y_min, y_max, resolution)
    x_grid, y_grid = np.meshgrid(x_lin, y_lin)

    try:
        points = df_clean[[x_col, y_col]].values
        values = df_clean[z_col].values
        z_grid = griddata(points, values, (x_grid, y_grid), method="linear")

        # Fill NaNs outside convex hull with nearest
        mask = np.isnan(z_grid)
        if np.any(mask):
            z_grid_nearest = griddata(points, values, (x_grid, y_grid), method="nearest")
            z_grid[mask] = z_grid_nearest[mask]

        return x_lin, y_lin, z_grid
    except Exception:
        return None, None, None


def build_combined_rsm_figure(df: pd.DataFrame) -> go.Figure:
    """Build a figure with 3 subplots (one per metric), each showing all 5 phi surfaces."""
    from plotly.subplots import make_subplots

    # Create 1 row, 3 columns of 3D subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        specs=[[{"type": "scene"}, {"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[label for _, label in METRICS],
        horizontal_spacing=0.02,
    )

    x_col, y_col = "rpm", "q_total"

    # Track traces per phi level for visibility toggling
    # traces_by_phi[phi_label] = list of trace indices
    traces_by_phi: dict[str, list[int]] = {label: [] for label in PHI_LEVELS.keys()}
    trace_idx = 0

    # Create surfaces for each metric (column) × phi (surface within column)
    for col_idx, (metric_col, metric_label) in enumerate(METRICS, start=1):
        if metric_col not in df.columns:
            continue

        colorscales = METRIC_COLORS.get(metric_col, ["Viridis"] * 5)

        for phi_idx, (phi_label, phi_val) in enumerate(PHI_LEVELS.items()):
            # Slice data for this phi level
            df_slice = df[np.isclose(df["phi"], phi_val, atol=0.05)].copy()
            if df_slice.empty:
                continue

            # Interpolate surface
            x_grid, y_grid, z_grid = interpolate_surface(df_slice, x_col, y_col, metric_col)
            if z_grid is None:
                continue

            trace_name = f"{phi_label}"
            # Show stoic by default
            visible = phi_val == 1.0

            surface = go.Surface(
                x=x_grid,
                y=y_grid,
                z=z_grid,
                name=trace_name,
                colorscale=colorscales[phi_idx % len(colorscales)],
                opacity=0.85,
                showscale=False,
                visible=visible,
                showlegend=(col_idx == 1),  # Only show legend for first column to avoid duplicates
                legendgroup=phi_label,  # Group by phi for unified legend toggle
                hovertemplate=(
                    f"<b>{metric_label}</b><br>"
                    f"<b>{phi_label}</b><br>"
                    f"RPM: %{{x:.0f}}<br>"
                    f"Q_total: %{{y:.0f}} J<br>"
                    f"Value: %{{z:.4f}}<extra></extra>"
                ),
            )
            fig.add_trace(surface, row=1, col=col_idx)
            traces_by_phi[phi_label].append(trace_idx)
            trace_idx += 1

    # Build dropdown menu buttons for toggling phi levels
    n_traces = trace_idx
    buttons = _build_phi_toggle_buttons(traces_by_phi, n_traces)

    # Configure each scene (3D subplot)
    scene_config = dict(
        xaxis_title="RPM",
        yaxis_title="Q_total (J)",
        zaxis_title="Response",
        camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
    )

    fig.update_layout(
        title=dict(
            text="Phase 1 Response Surfaces by Metric Type",
            font=dict(size=20),
            x=0.5,
        ),
        scene=scene_config,
        scene2=scene_config,
        scene3=scene_config,
        height=700,
        width=1600,
        margin=dict(l=0, r=0, b=50, t=100),
        updatemenus=buttons,
        legend=dict(
            title="Phi Level",
            yanchor="top",
            y=0.95,
            xanchor="left",
            x=1.01,
            tracegroupgap=5,
        ),
    )

    return fig


def _build_phi_toggle_buttons(
    traces_by_phi: dict[str, list[int]], n_traces: int
) -> list[dict[str, Any]]:
    """Build updatemenus buttons for toggling phi levels."""
    buttons_list = []

    # Show All
    buttons_list.append(
        dict(label="Show All Phi", method="update", args=[{"visible": [True] * n_traces}])
    )
    # Stoic Only (default)
    stoic_vis = [False] * n_traces
    for idx in traces_by_phi.get("Stoic (phi=1.0)", []):
        stoic_vis[idx] = True
    buttons_list.append(dict(label="Stoic Only", method="update", args=[{"visible": stoic_vis}]))
    # Hide All
    buttons_list.append(
        dict(label="Hide All", method="update", args=[{"visible": [False] * n_traces}])
    )

    # Individual phi toggles
    for phi_label, trace_indices in traces_by_phi.items():
        vis = [False] * n_traces
        for idx in trace_indices:
            vis[idx] = True
        short_label = phi_label.split("(")[1].rstrip(")")  # e.g., "phi=0.8"
        buttons_list.append(
            dict(label=f"Only {short_label}", method="update", args=[{"visible": vis}])
        )

    return [
        dict(
            type="dropdown",
            direction="down",
            showactive=True,
            x=0.0,
            xanchor="left",
            y=1.15,
            yanchor="top",
            buttons=buttons_list,
            bgcolor="white",
            bordercolor="#ccc",
            font=dict(size=12),
            pad=dict(l=10, r=10),
        )
    ]


def generate_combined_rsm_dashboard() -> None:
    """Main entry point to generate the combined RSM HTML."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    phase1_dir = os.path.join(base_dir, "goldens/phase1/doe_output")
    output_path = os.path.join(base_dir, "goldens", "RSM_COMBINED.html")

    raw_data = load_matrix_data(phase1_dir)
    if not raw_data:
        print("No Phase 1 DOE data found.")
        return

    df = pd.DataFrame(raw_data)
    print(f"Loaded {len(df)} rows from Phase 1 DOE.")

    fig = build_combined_rsm_figure(df)

    # Write HTML with Plotly JS included
    html_content = fig.to_html(full_html=True, include_plotlyjs="cdn")

    with open(output_path, "w") as f:
        f.write(html_content)

    print(f"Combined RSM dashboard generated: {output_path}")


if __name__ == "__main__":
    generate_combined_rsm_dashboard()
