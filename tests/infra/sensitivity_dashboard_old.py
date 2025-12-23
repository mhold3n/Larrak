import glob
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def load_matrix_data(matrix_dir: str) -> list[dict[str, Any]]:
    """
    Load all JSON result files from a matrix directory.
    Returns list of dicts.
    """
    data = []
    if not os.path.exists(matrix_dir):
        print(f"Warning: Directory not found: {matrix_dir}")
        return data

    json_files = glob.glob(os.path.join(matrix_dir, "*.json"))
    print(f"Loading {len(json_files)} files from {matrix_dir}...")

    for fpath in json_files:
        try:
            with open(fpath, "r") as f:
                res = json.load(f)

            item = {}
            # Flatten inputs
            inputs = res.get("inputs", {})
            for k, v in inputs.items():
                item[k] = v

            # Flatten output metrics
            output = res.get("output", {})
            status = output.get("status", "Failed")
            obj_val = output.get("objective", float("nan"))

            item["status"] = status
            item["objective"] = obj_val
            item["iter_count"] = output.get("iter_count", 0)

            # Trajectory stats (Impulse/Shock)
            traj = output.get("trajectories", {})

            if "acc" in traj:
                acc = np.array(traj["acc"])
                item["peak_accel"] = np.max(np.abs(acc)) if len(acc) > 0 else 0.0
            elif "a" in traj:  # Phase 3 naming
                acc = np.array(traj["a"])
                item["peak_accel"] = np.max(np.abs(acc)) if len(acc) > 0 else 0.0
            else:
                item["peak_accel"] = 0.0

            # Phase 3 specific: cam radius (size)
            if "r" in traj:
                r = np.array(traj["r"])
                item["min_r"] = np.min(r) if len(r) > 0 else 0.0

            data.append(item)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")

    return data


def create_3d_surface(
    df: pd.DataFrame, x_col: str, y_col: str, z_col: str, title: str
) -> go.Surface:
    """
    Create a 3D surface plot trace.
    Requires data to be pivoted into a grid.
    """
    # Pivot to grid
    pivot = df.pivot_table(values=z_col, index=y_col, columns=x_col)

    x_vals = pivot.columns.values
    y_vals = pivot.index.values
    z_vals = pivot.values

    surface = go.Surface(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        colorscale="Viridis",
        colorbar=dict(title=z_col),
        name=title,
    )
    return surface


def calculate_sensitivity(
    df: pd.DataFrame, x_col: str, y_col: str, z_col: str
) -> Optional[pd.DataFrame]:
    """
    Calculate magnitude of gradient (sensitivity) of z wrt x and y.
    Returns DataFrame with 'sensitivity' column added.
    """
    try:
        pivot = df.pivot_table(values=z_col, index=y_col, columns=x_col)
        grad_y, grad_x = np.gradient(pivot.values, pivot.index.values, pivot.columns.values)

        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        sens_df = pd.DataFrame(magnitude, index=pivot.index, columns=pivot.columns)
        return sens_df
    except Exception as e:
        print(f"Sensitivity calc failed: {e}")
        return None


def _get_axis_from_tag(tag: str, df_cols: list[str]) -> Optional[str]:
    """Helper to map name tag to dataframe column."""
    if tag == "rpm" and "rpm" in df_cols:
        return "rpm"
    if tag == "load":
        if "q_total" in df_cols:
            return "q_total"
        if "load_scale" in df_cols:
            return "load_scale"
    if tag == "boost" and "p_boost_bars" in df_cols:
        return "p_boost_bars"
    if tag == "stoic" and "phi" in df_cols:
        return "phi"
    if tag == "size" and "r_max" in df_cols:
        return "r_max"
    return None


def get_matrix_axes(df: pd.DataFrame, matrix_name: str) -> tuple[str | None, str | None]:
    """Determine X and Y axes based on matrix name or dataframe variability."""
    if "_x_" in matrix_name:
        parts = matrix_name.split("_x_")
        ax1_tag = parts[0]
        ax2_tag = parts[1]

        x_col = _get_axis_from_tag(ax1_tag, list(df.columns))
        y_col = _get_axis_from_tag(ax2_tag, list(df.columns))

        if x_col and y_col:
            return x_col, y_col

    # Fallback
    exclude = [
        "status",
        "objective",
        "iter_count",
        "peak_accel",
        "min_r",
        "trajectories",
        "omega",
        "work_j",
        "efficiency",
        "file",
        "Thermal Efficiency",
        "status_code",
    ]
    potential_axes = [c for c in df.columns if c not in exclude]
    axes = [c for c in potential_axes if df[c].nunique() > 1]

    if len(axes) == 2:
        return axes[0], axes[1]
    return None, None


def _create_solver_status_plot(
    df: pd.DataFrame, x_col: str, y_col: str, matrix_name: str
) -> go.Figure:
    """Create a heatmap for solver status."""

    def status_to_int(s: str) -> float:
        if s in ["Optimal", "Solve_Succeeded"]:
            return 1.0
        if "Acceptable" in s:
            return 0.5
        return 0.0

    df["status_code"] = df["status"].apply(status_to_int)
    status_pivot = df.pivot_table(values="status_code", index=y_col, columns=x_col)

    fig = go.Figure(
        data=go.Heatmap(
            z=status_pivot.values,
            x=sorted(df[x_col].unique()),
            y=sorted(df[y_col].unique()),
            colorscale=[[0, "red"], [0.5, "yellow"], [1.0, "green"]],
            colorbar=dict(
                title="Solver Status (1=Opt, 0=Fail)",
                tickvals=[0, 0.5, 1],
                ticktext=["Fail", "Accept", "Optimal"],
            ),
        )
    )
    fig.update_layout(
        title=f"{matrix_name}: Solver Stability Map",
        xaxis_title=x_col,
        yaxis_title=y_col,
        width=800,
        height=600,
    )
    return fig


def _create_sensitivity_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    z_col: str,
    z_title: str,
    phase_name: str,
    matrix_name: str,
) -> go.Figure:
    """Create heatmap for sensitivity or impulse."""
    heatmap_z = None
    heatmap_title = "Peak Impulse"

    pivot_for_axes = df.pivot_table(values=z_col, index=y_col, columns=x_col)

    if "phase1" in phase_name.lower() or "rpm_x_load" in matrix_name:
        pivot_z = df.pivot_table(values=z_col, index=y_col, columns=x_col)
        if not pivot_z.empty and pivot_z.shape[0] > 1 and pivot_z.shape[1] > 1:
            grad_y, grad_x = np.gradient(
                pivot_z.values, pivot_z.index.values, pivot_z.columns.values
            )
            x_grid, y_grid = np.meshgrid(pivot_z.columns.values, pivot_z.index.values)
            z_grid = pivot_z.values

            with np.errstate(divide="ignore", invalid="ignore"):
                e_x = grad_x * (x_grid / (z_grid + 1e-9))
                e_y = grad_y * (y_grid / (z_grid + 1e-9))
                elast_mag = np.sqrt(e_x**2 + e_y**2)
                elast_mag = np.nan_to_num(elast_mag)

            heatmap_z = elast_mag
            heatmap_title = f"{z_title} Elasticity (Relative Sensitivity)"

    if heatmap_z is None:
        pivot_imp = df.pivot_table(values="peak_accel", index=y_col, columns=x_col)
        heatmap_z = pivot_imp.values

    heatmap = go.Heatmap(
        z=heatmap_z,
        x=pivot_for_axes.columns,
        y=pivot_for_axes.index,
        colorscale="Viridis" if "Sensitivity" in heatmap_title else "Magma",
        colorbar=dict(title=heatmap_title),
    )
    fig = go.Figure(data=heatmap)
    fig.update_layout(
        title=f"{matrix_name}: {heatmap_title}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        width=800,
        height=600,
    )
    return fig


def process_matrix_figures(matrix_dir: str, matrix_name: str, phase_name: str) -> list[go.Figure]:
    """Generate figures for a single matrix."""
    data = load_matrix_data(matrix_dir)
    if not data:
        return []

    df = pd.DataFrame(data)
    figs: list[go.Figure] = []

    x_col, y_col = get_matrix_axes(df, matrix_name)

    if not x_col or not y_col:
        print(f"Skipping {matrix_name}: Could not identify 2 varying axes.")
        return []

    z_col = "objective"
    z_title = "Objective"

    # Metric Transformation
    if "phase1" in phase_name.lower() or "rpm_x_load" in matrix_name:
        if "q_total" in df.columns:
            df["q_total"] = pd.to_numeric(df["q_total"])
            df["efficiency"] = -1.0 * df["objective"] / (df["q_total"] + 1e-9)
            z_col = "efficiency"
            z_title = "Thermal Efficiency"
    elif "phase3" in phase_name.lower():
        z_title = "Tracking Cost (Error)"
        z_col = "objective"

    # 1. Primary Surface
    surface = go.Surface(
        z=df.pivot_table(values=z_col, index=y_col, columns=x_col).values,
        x=sorted(df[x_col].unique()),
        y=sorted(df[y_col].unique()),
        colorscale="Viridis",
        colorbar=dict(title=z_title),
    )
    fig = go.Figure(data=[surface])
    fig.update_layout(
        title=f"{matrix_name}: {z_title} Surface",
        scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_title),
        width=800,
        height=600,
    )
    figs.append(fig)

    # 2. Solver Stability Map
    figs.append(_create_solver_status_plot(df, x_col, y_col, matrix_name))

    # 3. Iteration Map
    if "iter_count" in df.columns:
        iter_pivot = df.pivot_table(values="iter_count", index=y_col, columns=x_col)
        fig_iter = go.Figure(
            data=go.Heatmap(
                z=iter_pivot.values,
                x=sorted(df[x_col].unique()),
                y=sorted(df[y_col].unique()),
                colorscale="Hot",
                reversescale=True,
                colorbar=dict(title="Iterations"),
            )
        )
        fig_iter.update_layout(
            title=f"{matrix_name}: Convergence Difficulty (Iterations)",
            xaxis_title=x_col,
            yaxis_title=y_col,
            width=800,
            height=600,
        )
        figs.append(fig_iter)

    # 4. Sensitivity
    figs.append(_create_sensitivity_plot(df, x_col, y_col, z_col, z_title, phase_name, matrix_name))

    return figs


def generate_dashboard() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    phases = [
        (
            "Phase 1 (Thermo)",
            os.path.join(base_dir, "goldens/phase1/matrix_output"),
            "SENSITIVITY_REPORT_PHASE1.html",
        ),
        (
            "Phase 3 (Mech)",
            os.path.join(base_dir, "goldens/phase3/matrix_output"),
            "SENSITIVITY_REPORT_PHASE3.html",
        ),
    ]

    for phase_name, phase_dir, output_filename in phases:
        if not os.path.exists(phase_dir):
            continue

        phase_figs = []
        subdirs = [d for d in os.listdir(phase_dir) if os.path.isdir(os.path.join(phase_dir, d))]

        for subdir in subdirs:
            if subdir in ["plots", "composite_plots"]:
                continue

            matrix_path = os.path.join(phase_dir, subdir)
            figs = process_matrix_figures(matrix_path, subdir, phase_name)
            phase_figs.extend(figs)

        output_path = os.path.join(base_dir, "goldens", output_filename)
        with open(output_path, "w") as f:
            f.write("<html><head><title>Sensitivity Report</title></head><body>")
            f.write(f"<h1>{phase_name} Sensitivity & Impulse Report</h1>")
            for i, fig in enumerate(phase_figs):
                f.write(
                    f"<div>{fig.to_html(full_html=False, include_plotlyjs='cdn' if i == 0 else False)}</div>"
                )
                f.write("<hr>")
            f.write("</body></html>")

        print(f"Dashboard generated at: {output_path}")


if __name__ == "__main__":
    generate_dashboard()
