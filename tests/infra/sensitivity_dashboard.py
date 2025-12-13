import glob
import json
import os
import sys
from typing import Any, TextIO

# Add project root to sys.path for direct execution
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore

from tests.infra.interpretation import DOEAnalyzer


def load_matrix_data(matrix_dir: str) -> list[dict[str, Any]]:
    """
    Load data. Supports legacy (folder of JSONs) and new (single CSV/Parquet) formats.
    """
    data: list[dict[str, Any]] = []
    if not os.path.exists(matrix_dir):
        print(f"Warning: Directory not found: {matrix_dir}")
        return data

    # 1. Check for DOE CSVs
    csv_files = glob.glob(os.path.join(matrix_dir, "*.csv"))
    if csv_files:
        # Load the most recent CSV
        latest_csv = max(csv_files, key=os.path.getmtime)
        print(f"Loading DOE data from {latest_csv}...")
        df = pd.read_csv(latest_csv)
        # Ensure we don't have NaNs in critical columns
        df = df.replace({np.nan: None})
        records: list[dict[str, Any]] = df.to_dict("records")  # type: ignore[assignment]
        return records

    # 2. Legacy JSON loading (Fallback)
    json_files = glob.glob(os.path.join(matrix_dir, "*.json"))
    if json_files:
        print(f"Loading {len(json_files)} JSON files from {matrix_dir}...")
        for fpath in json_files:
            try:
                with open(fpath) as f:
                    res = json.load(f)

                # Basic parsing mimicking previous logic
                item = {}
                inputs = res.get("inputs", {})
                for k, v in inputs.items():
                    item[k] = v
                output = res.get("output", {})
                item["status"] = output.get("status", "Failed")
                item["objective"] = output.get("objective", float("nan"))
                item["iter_count"] = output.get("iter_count", 0)

                traj = output.get("trajectories", {})
                if "acc" in traj:
                    acc = np.array(traj["acc"])
                    item["peak_accel"] = np.max(np.abs(acc)) if len(acc) > 0 else 0.0
                elif "a" in traj:
                    acc = np.array(traj["a"])
                    item["peak_accel"] = np.max(np.abs(acc)) if len(acc) > 0 else 0.0
                else:
                    item["peak_accel"] = 0.0

                if "r" in traj:
                    r = np.array(traj["r"])
                    item["min_r"] = np.min(r) if len(r) > 0 else 0.0
                elif "r_max" in inputs:
                    item["r_max"] = inputs["r_max"]

                data.append(item)
            except Exception:
                pass
        return data
    return []


# --- Visualization Builders ---


def create_pareto_chart(
    effects_df: pd.DataFrame, title: str = "Pareto Chart of Standardized Effects"
) -> go.Figure:
    """A. Structure Layer: Pareto Chart"""
    df_sorted = effects_df.sort_values("AbsEffect", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_sorted["AbsEffect"],
            y=df_sorted["Term"],
            orientation="h",
            marker_color="rgb(55, 83, 109)",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="Standardized Effect / t-value",
        yaxis_title="Term",
        height=400,
        margin=dict(l=150),
    )
    # Add threshold line? (Maybe later)
    return fig


def create_interaction_plot(analyzer: DOEAnalyzer, factor1: str, factor2: str) -> go.Figure:
    """A. Structure Layer: Interaction Plot"""
    grid = analyzer.get_interaction_grid(factor1, factor2, resolution=10)

    # We want line plots: X=Factor1, Y=Prediction, Lines=Factor2 Levels
    # Bin factor2 into a few discrete levels for plotting lines
    levels = np.sort(grid[factor2].unique())
    # Downsample to 3-4 lines if too many
    if len(levels) > 5:
        indices = np.linspace(0, len(levels) - 1, 4, dtype=int)
        levels = levels[indices]

    fig = go.Figure()

    for lvl in levels:
        subset = grid[grid[factor2] == lvl].sort_values(factor1)
        fig.add_trace(
            go.Scatter(
                x=subset[factor1],
                y=subset["Prediction"],
                mode="lines+markers",
                name=f"{factor2}={lvl:.2g}",
            )
        )

    fig.update_layout(
        title=f"Interaction: {factor1} x {factor2}",
        xaxis_title=factor1,
        yaxis_title="Response",
        height=400,
    )
    return fig


def create_tornado_plot(sens_df: pd.DataFrame) -> go.Figure:
    """B. Influence Layer: Tornado Plot"""
    # Sort by sensitivity
    df_sorted = sens_df.sort_values("SensitivityScore", ascending=True)

    fig = go.Figure(
        go.Bar(
            x=df_sorted["SensitivityScore"],
            y=df_sorted["Factor"],
            orientation="h",
            marker_color="teal",
            text=[f"{v:.1f}%" for v in df_sorted["Relative %"]],
            textposition="auto",
        )
    )
    fig.update_layout(
        title="Global Sensitivity (Tornado)",
        xaxis_title="Sensitivity Index (Sobol proxy)",
        height=400,
    )
    return fig


def create_local_sensitivity_heatmap(
    df: pd.DataFrame, x_col: str, y_col: str, z_col: str
) -> go.Figure:
    """B. Influence Layer: Local Sensitivity Heatmap (Elasticity)"""
    # Requires pivoting
    pivot = df.pivot_table(values=z_col, index=y_col, columns=x_col)
    if pivot.empty or pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return go.Figure()

    grad_y, grad_x = np.gradient(pivot.values, pivot.index.values, pivot.columns.values)

    # Calculate magnitude
    mag = np.sqrt(grad_x**2 + grad_y**2)

    fig = go.Figure(
        go.Heatmap(
            z=mag,
            x=pivot.columns,
            y=pivot.index,
            colorscale="Hot",
            colorbar=dict(title="|Gradient|"),
        )
    )
    fig.update_layout(
        title=f"Local Sensitivity: d({z_col})/d(Params)",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=500,
    )
    return fig


def create_residual_plot(residuals_df: pd.DataFrame) -> go.Figure:
    """C. Shape Layer: Residuals vs Fitted"""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=residuals_df["Predicted"],
            y=residuals_df["Residual"],
            mode="markers",
            marker=dict(color="red", opacity=0.6),
        )
    )
    fig.add_shape(
        type="line",
        x0=residuals_df["Predicted"].min(),
        y0=0,
        x1=residuals_df["Predicted"].max(),
        y1=0,
        line=dict(color="black", dash="dash"),
    )
    fig.update_layout(
        title="Residuals vs Fitted (Check for Randomness)",
        xaxis_title="Fitted Value",
        yaxis_title="Residual",
        height=400,
    )
    return fig


def create_response_surface(
    df: pd.DataFrame, x_col: str, y_col: str, z_col: str, analyzer: DOEAnalyzer | None = None
) -> go.Figure:
    """C. Shape Layer: 3D Surface with Raw Scatter Overlay"""

    # 1. Prediction Surface (Smoothed)
    # Priority: Analyzer Model -> Interpolation -> Pivot -> Error
    x_grid = None
    y_grid = None
    z_grid = None
    error_msg = None

    # Resolution for grid
    resolution = 25
    x_min, x_max = df[x_col].min(), df[x_col].max()
    y_min, y_max = df[y_col].min(), df[y_col].max()
    x_lin = np.linspace(x_min, x_max, resolution)
    y_lin = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_lin, y_lin)

    # Attempt 1: Analyzer Model
    if analyzer:
        try:
            # We need to hold other variables constant (at mean/mode)
            base_row = df.mean(numeric_only=True).to_frame().T
            grid_df = pd.DataFrame(
                np.tile(base_row.values, (resolution * resolution, 1)), columns=base_row.columns
            )
            grid_df[x_col] = X.ravel()
            grid_df[y_col] = Y.ravel()

            Z_flat = analyzer.predict(grid_df)
            z_grid = Z_flat.reshape(resolution, resolution)
            x_grid = x_lin
            y_grid = y_lin
        except Exception:
            pass

    # Attempt 2: Scipy Interpolation (Linear/RBF) fallback
    if z_grid is None:
        try:
            from scipy.interpolate import griddata

            # Drop NaNs
            df_n = df.dropna(subset=[x_col, y_col, z_col])
            points = df_n[[x_col, y_col]].values
            values = df_n[z_col].values

            # Linear is robust but maybe jagged. Cubic is smoother but can overshoot.
            # Let's try linear for guaranteed continuity within convex hull
            z_grid = griddata(points, values, (X, Y), method="linear")

            # Fill NaNs (outside convex hull) with nearest to allow a full surface?
            # Or just leave them transparent. Transparent is usually better/truthful.
            # If we strictly want a continuous surface filling the box, we can use 'nearest' to fill gaps
            mask = np.isnan(z_grid)
            if np.any(mask):
                z_grid_nearest = griddata(points, values, (X, Y), method="nearest")
                z_grid[mask] = z_grid_nearest[mask]

            x_grid = x_lin
            y_grid = y_lin
        except Exception as e:
            error_msg = f"Interpolation failed: {str(e)}"

    data = []

    # 2. Raw Scatter Points (Always show)
    scatter = go.Scatter3d(
        x=df[x_col],
        y=df[y_col],
        z=df[z_col],
        mode="markers",
        marker=dict(size=4, color="black", symbol="circle", opacity=0.8),
        name="Raw Data",
    )
    data.append(scatter)

    # 1. Surface (If available)
    if z_grid is not None and z_grid.size > 0:
        surface = go.Surface(
            z=z_grid,
            x=x_grid,
            y=y_grid,
            colorscale="Viridis",
            colorbar=dict(title=z_col),
            opacity=0.8,
            name="Response Surface",
        )
        data.append(surface)
    elif error_msg:
        # Add error annotation in layout, not data
        pass

    fig = go.Figure(data=data)

    layout_args = dict(
        title=f"Response Surface: {z_col}",
        scene=dict(xaxis_title=x_col, yaxis_title=y_col, zaxis_title=z_col),
        height=600,
        width=800,
        margin=dict(l=0, r=0, b=0, t=40),
    )

    if error_msg or z_grid is None:
        layout_args["annotations"] = [
            dict(
                text=error_msg if error_msg else "Could not fit surface",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=20, color="red"),
            )
        ]

    fig.update_layout(**layout_args)
    return fig


def create_contour_plot(df: pd.DataFrame, x_col: str, y_col: str, z_col: str) -> go.Figure:
    """C. Shape Layer: Contour Plot"""

    # Needs valid grid for contour. Use same interpolation logic.
    z_grid = None
    x_grid = None
    y_grid = None
    error_msg = None

    try:
        from scipy.interpolate import griddata

        resolution = 100
        x_min, x_max = df[x_col].min(), df[x_col].max()
        y_min, y_max = df[y_col].min(), df[y_col].max()
        x_lin = np.linspace(x_min, x_max, resolution)
        y_lin = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_lin, y_lin)

        df_n = df.dropna(subset=[x_col, y_col, z_col])
        points = df_n[[x_col, y_col]].values
        values = df_n[z_col].values

        z_grid = griddata(points, values, (X, Y), method="linear")

        # Fill edges
        mask = np.isnan(z_grid)
        if np.any(mask):
            z_grid_nearest = griddata(points, values, (X, Y), method="nearest")
            z_grid[mask] = z_grid_nearest[mask]

        x_grid = x_lin
        y_grid = y_lin
    except Exception as e:
        error_msg = str(e)

    fig = go.Figure()

    if z_grid is not None:
        fig.add_trace(
            go.Contour(
                z=z_grid,
                x=x_grid,
                y=y_grid,
                colorscale="Viridis",
                contours=dict(showlabels=True, labelfont=dict(size=12, color="white")),
                colorbar=dict(title=z_col),
            )
        )
        # Overlay raw points
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(size=8, color="black", opacity=0.5, line=dict(width=1, color="white")),
                name="Points",
            )
        )
    else:
        # Fallback scatter only
        fig.add_trace(
            go.Scatter(
                x=df[x_col],
                y=df[y_col],
                mode="markers",
                marker=dict(size=10, color=df[z_col], colorscale="Viridis", showscale=True),
                text=df[z_col].apply(lambda v: f"{v:.3f}"),
            )
        )
        fig.add_annotation(
            text=f"Surface Error: {error_msg}" if error_msg else "Cannot interpolate surface",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(color="red"),
        )

    fig.update_layout(
        title=f"Contour Map: {z_col}",
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=500,
    )
    return fig


def generate_phase_report(
    df: pd.DataFrame, phase_name: str, override_response: str | None = None
) -> dict[str, Any]:
    """Generates all artifacts for a single phase layer."""

    # 1. Config based on Phase
    inputs, response, title_z = _get_phase_config(df, phase_name, override_response)

    available_inputs = [c for c in inputs if c in df.columns and df[c].nunique() > 1]

    if len(available_inputs) < 2 or response not in df.columns:
        return {"error": "Insufficient data"}

    # Sanitize Data
    df_clean = _clean_dataframe(df, available_inputs, response)

    # 2. Analysis
    analyzer = DOEAnalyzer(df=df_clean)
    fit_success = _try_fit_model(analyzer, available_inputs, response, len(df_clean))

    # A. Structure Layer (Requires Fit)
    pareto_fig, interaction_fig, base_factors = _generate_structure_figures(
        analyzer, fit_success, available_inputs
    )

    # B. Influence Layer (Requires Fit)
    tornado_fig, local_sens_fig = _generate_influence_figures(
        analyzer, fit_success, df, available_inputs, base_factors, response
    )

    # C. Shape Layer
    resid_fig, surface_3d, contour_map = _generate_shape_figures(
        analyzer, fit_success, df, available_inputs, base_factors, response
    )

    # D. Physics / Tables
    anova_html = _generate_anova_table(analyzer, fit_success)

    return {
        "structure": [pareto_fig, interaction_fig],
        "influence": [tornado_fig, local_sens_fig],
        "shape": [surface_3d, contour_map, resid_fig],
        "tables": {"anova": anova_html},
        "meta": {"response": title_z},
    }


def _get_phase_config(
    df: pd.DataFrame, phase_name: str, override_response: str | None
) -> tuple[list[str], str, str]:
    """Determine inputs and response variable based on phase."""
    name_clean = phase_name.lower()
    if "phase 1" in name_clean:
        return _get_phase1_config(df, override_response)
    if "phase 2" in name_clean:
        inputs = ["mean_ratio", "stroke", "conrod"]
        response = override_response if override_response else "fit_error"
        title_z = "Fitting Error" if not override_response else override_response
        return inputs, response, title_z
    if "phase 3" in name_clean:
        inputs = ["r_max", "load_scale", "rpm"]
        response = override_response if override_response else "objective"
        title_z = "Tracking Cost" if not override_response else override_response
        return inputs, response, title_z
    return [], "objective", "Objective"


def _get_phase1_config(
    df: pd.DataFrame, override_response: str | None
) -> tuple[list[str], str, str]:
    # Support both legacy and new column names
    possible_inputs = ["rpm", "q_total_j", "phi_est", "p_int_bar", "fuel_mass_mg"]
    found_inputs = [c for c in possible_inputs if c in df.columns]

    # Prefer physical inputs if available
    if "p_int_bar" in found_inputs and "fuel_mass_mg" in found_inputs:
        inputs = ["rpm", "p_int_bar", "fuel_mass_mg"]
    else:
        inputs = ["rpm", "q_total", "phi"]

    if override_response:
        response = override_response
        title_z = override_response.replace("_", " ").title()
    elif "thermal_efficiency" in df.columns:
        response = "thermal_efficiency"
        title_z = "Thermal Efficiency"
    elif "gas_kinetic_efficiency" in df.columns:
        response = "gas_kinetic_efficiency"
        title_z = "Gas-Kinetic Efficiency"
    else:
        response = "objective"
        title_z = "Response"
    return inputs, response, title_z


def _clean_dataframe(df: pd.DataFrame, cols: list[str], response: str) -> pd.DataFrame:
    """Remove NaN/Inf rows."""
    cols_to_check = [*cols, response]
    df_clean = df.dropna(subset=cols_to_check).copy()
    return df_clean[~df_clean[cols_to_check].isin([np.inf, -np.inf]).any(axis=1)]


def _try_fit_model(analyzer: DOEAnalyzer, inputs: list[str], response: str, n_samples: int) -> bool:
    if n_samples < 5:
        return False
    try:
        analyzer.fit_model(inputs, response)
        return True
    except Exception:
        return False


def _generate_structure_figures(
    analyzer: DOEAnalyzer, fit_success: bool, available_inputs: list[str]
) -> tuple[go.Figure, go.Figure, list[str]]:
    pareto_fig = go.Figure()
    interaction_fig = go.Figure()
    base_factors = available_inputs[:2] if len(available_inputs) >= 2 else ["x", "y"]

    if fit_success:
        try:
            effects_df = analyzer.get_standardized_effects()
            pareto_fig = create_pareto_chart(effects_df)

            top_factors = (
                effects_df.sort_values("AbsEffect", ascending=False)["Term"].head(2).tolist()
            )
            base_factors_fit = [f for f in available_inputs if f in top_factors]
            if len(base_factors_fit) >= 2:
                base_factors = base_factors_fit

            interaction_fig = create_interaction_plot(analyzer, base_factors[0], base_factors[1])
        except Exception:
            pass
    return pareto_fig, interaction_fig, base_factors


def _generate_influence_figures(
    analyzer: DOEAnalyzer,
    fit_success: bool,
    df: pd.DataFrame,
    available_inputs: list[str],
    base_factors: list[str],
    response: str,
) -> tuple[go.Figure, go.Figure]:
    tornado_fig = go.Figure()
    local_sens_fig = go.Figure()

    if fit_success:
        bounds = {c: (df[c].min(), df[c].max()) for c in available_inputs}
        sens_df = analyzer.compute_sensitivity(bounds)
        tornado_fig = create_tornado_plot(sens_df)

        slice_df = df.copy()
        if len(available_inputs) > 2:
            third_dim = [c for c in available_inputs if c not in base_factors][0]
            mode_val = df[third_dim].mode()[0]
            slice_df = slice_df[slice_df[third_dim] == mode_val]

        local_sens_fig = create_local_sensitivity_heatmap(
            slice_df, base_factors[0], base_factors[1], response
        )
    return tornado_fig, local_sens_fig


def _generate_shape_figures(
    analyzer: DOEAnalyzer,
    fit_success: bool,
    df: pd.DataFrame,
    available_inputs: list[str],
    base_factors: list[str],
    response: str,
) -> tuple[go.Figure, go.Figure, go.Figure]:
    resid_fig = go.Figure()
    surface_3d = go.Figure()
    contour_map = go.Figure()
    slice_df = df.copy()  # Fallback

    if fit_success:
        resid_df = analyzer.calculate_residuals()
        resid_fig = create_residual_plot(resid_df)
        surface_3d = create_response_surface(
            slice_df, base_factors[0], base_factors[1], response, analyzer
        )
        contour_map = create_contour_plot(slice_df, base_factors[0], base_factors[1], response)
    elif len(slice_df) > 0 and len(base_factors) >= 2:
        surface_3d = create_response_surface(
            slice_df, base_factors[0], base_factors[1], response, None
        )
        contour_map = create_contour_plot(slice_df, base_factors[0], base_factors[1], response)

    return resid_fig, surface_3d, contour_map


def _generate_anova_table(analyzer: DOEAnalyzer, fit_success: bool) -> str:
    if not fit_success:
        return "<p>Data insufficient or fit failed for ANOVA</p>"
    try:
        anova_df = analyzer.run_anova()
        return str(anova_df.to_html(classes="datatable"))
    except Exception as e:
        return f"<p>ANOVA failed: {e!s}</p>"


def generate_dashboard() -> None:
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(base_dir, "goldens", "SENSITIVITY_REPORT_FULL.html")

    phases = [
        ("Phase 1 (Thermo)", os.path.join(base_dir, "goldens/phase1/doe_output")),
        ("Phase 2 (Interpreter)", os.path.join(base_dir, "goldens/phase2/doe_output")),
        ("Phase 3 (Mech)", os.path.join(base_dir, "goldens/phase3/doe_output")),
    ]

    # Pre-load all reports
    reports: dict[str, dict[str, Any]] = {}
    phase1_slices: dict[str, pd.DataFrame] = {}  # Store sub-reports for Phase 1 (Rich, Stoic, etc)

    phi_levels = {
        "Rich (phi=1.2)": 1.2,
        "Slightly Rich (phi=1.1)": 1.1,
        "Stoic (phi=1.0)": 1.0,
        "Slightly Lean (phi=0.9)": 0.9,
        "Lean (phi=0.8)": 0.8,
    }

    for name, pdir in phases:
        raw_data = load_matrix_data(pdir)
        if not raw_data:
            continue

        df = pd.DataFrame(raw_data)

        if "Phase 1" in name:
            _append_phase1_report(df, name, reports, phase1_slices, phi_levels)
        else:
            reports[name] = generate_phase_report(df, name)

    with open(output_path, "w") as f:
        _write_dashboard_html(f, phases, reports, phase1_slices, phi_levels)

    print(f"Dashboard generated: {output_path}")


def _append_phase1_report(
    df: pd.DataFrame,
    name: str,
    reports: dict[str, Any],
    phase1_slices: dict[str, pd.DataFrame],
    phi_levels: dict[str, float],
) -> None:
    phi_col = "phi_est" if "phi_est" in df.columns else "phi"

    if phi_col in df.columns:
        for label, phi_val in phi_levels.items():
            # Use wider tolerance for physical DOE where phi is calculated
            df_slice = df[np.isclose(df[phi_col], phi_val, atol=0.1)].copy()
            if not df_slice.empty:
                phase1_slices[label] = df_slice
    reports[name] = generate_phase_report(df, name)


def _write_dashboard_html(
    f: Any,
    phases: list[tuple[str, str]],
    reports: dict[str, Any],
    phase1_slices: dict[str, pd.DataFrame],
    phi_levels: dict[str, float],
) -> None:
    _write_header(f)
    phase_names = [p[0] for p in phases if p[0] in reports]
    _write_nav_tabs(f, phase_names)
    _write_tab_contents(f, phase_names, reports, phase1_slices, phi_levels)
    _write_footer(f)


def _write_header(f: Any) -> None:
    f.write("""
    <html>
    <head>
        <title>Optimization Sensitivity Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-3.1.1.min.js"></script>
        <style>
            body { font-family: "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f7f9fa; color: #333; }
            .header { background: #fff; padding: 20px; border-bottom: 1px solid #ddd; text-align: center; }
            .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
            .tab-nav { display: flex; border-bottom: 2px solid #e1e4e8; margin-bottom: 20px; flex-wrap: wrap;}
            .tab-btn { padding: 12px 24px; cursor: pointer; border: none; background: none; font-size: 16px; font-weight: 600; color: #586069; border-bottom: 2px solid transparent; margin-bottom: -2px; }
            .tab-btn:hover { color: #0366d6; }
            .tab-btn.active { color: #0366d6; border-bottom: 2px solid #0366d6; }
            .sub-nav .tab-btn { font-size: 14px; padding: 8px 16px; color: #666; }
            .sub-nav .tab-btn.active { color: #d73a49; border-bottom-color: #d73a49; }
            .tab-content { display: none; width: 100%; }
            .tab-content.active { display: block; }
            .layer { background: #fff; border-radius: 6px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 30px; padding: 20px; }
            .layer-title { font-size: 18px; font-weight: 600; color: #24292e; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-bottom: 20px; display: flex; align-items: center; }
            .layer-badge { background: #e1e4e8; color: #586069; font-size: 12px; padding: 2px 8px; border-radius: 12px; margin-right: 10px; }
            .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }
            .full-width { grid-column: 1 / -1; }
            .datatable { width: 100%; border-collapse: collapse; font-size: 14px; }
            .datatable th { background: #f6f8fa; border: 1px solid #e1e4e8; padding: 8px; text-align: left; }
            .datatable td { border: 1px solid #e1e4e8; padding: 8px; }
            .error { color: red; padding: 20px; }
        </style>
        <script>
            function openTab(evt, contentClass, tabId) {
                var i, content, btns;
                content = document.getElementsByClassName(contentClass);
                for (i = 0; i < content.length; i++) {
                    content[i].classList.remove("active");
                }
                var btnContainer = evt.currentTarget.parentNode;
                btns = btnContainer.getElementsByClassName("tab-btn");
                for (i = 0; i < btns.length; i++) {
                    btns[i].classList.remove("active");
                }
                document.getElementById(tabId).classList.add("active");
                evt.currentTarget.classList.add("active");
            }
        </script>
    </head>
    <body>
        <div class="header">
            <h1>Sensitivity & Calibration Dashboard</h1>
            <p>Advanced Diagnostics for Design of Experiments</p>
        </div>
        <div class="container">
            <div class="tab-nav">
    """)


def _write_nav_tabs(f: Any, phase_names: list[str]) -> None:
    first_main = True
    for name in phase_names:
        safe_id = "main_" + name.replace(" ", "_").replace("(", "").replace(")", "")
        active = " active" if first_main else ""
        f.write(
            f"<button class=\"tab-btn{active}\" onclick=\"openTab(event, 'main-content', '{safe_id}')\">{name}</button>"
        )
        first_main = False
    f.write("</div>")


def _write_tab_contents(
    f: Any,
    phase_names: list[str],
    reports: dict[str, Any],
    phase1_slices: dict[str, pd.DataFrame],
    phi_levels: dict[str, float],
) -> None:
    first_main = True
    for name in phase_names:
        safe_id = "main_" + name.replace(" ", "_").replace("(", "").replace(")", "")
        active_cls = " active" if first_main else ""
        f.write(f'<div id="{safe_id}" class="tab-content main-content{active_cls}">')

        if "Phase 1" in name and phase1_slices:
            _write_phase1_subtabs(f, safe_id, phase1_slices, phi_levels, name)
        elif name in reports:
            render_report_body(f, reports[name])

        f.write("</div>")
        first_main = False


def _write_phase1_subtabs(
    f: Any,
    safe_id: str,
    phase1_slices: dict[str, pd.DataFrame],
    phi_levels: dict[str, float],
    name: str,
) -> None:
    f.write('<div class="tab-nav sub-nav">')
    sub_first = True
    for sub_label in phi_levels.keys():
        if sub_label in phase1_slices:
            # Use full label for safety, replacing regex-like chars
            safe_label = (
                sub_label.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(".", "p")
                .replace("=", "_")
            )
            sub_safe_id = safe_id + "_" + safe_label
            sub_active = " active" if sub_first else ""
            f.write(
                f"<button class=\"tab-btn{sub_active}\" onclick=\"openTab(event, 'sub-content-p1', '{sub_safe_id}')\">{sub_label}</button>"
            )
            sub_first = False
    f.write("</div>")

    sub_first = True
    p1_metrics = [
        ("gas_kinetic_efficiency", "Gas-Kinetic Efficiency"),
        ("thermal_efficiency", "Thermal Efficiency"),
        ("peak_pressure_bar", "Peak Pressure (bar)"),
    ]

    for sub_label in phi_levels.keys():
        if sub_label in phase1_slices:
            df_slice = phase1_slices[sub_label]
            safe_label = (
                sub_label.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace(".", "p")
                .replace("=", "_")
            )
            sub_safe_id = safe_id + "_" + safe_label
            sub_active = " active" if sub_first else ""

            f.write(f'<div id="{sub_safe_id}" class="tab-content sub-content-p1{sub_active}">')

            for m_col, m_label in p1_metrics:
                if m_col not in df_slice.columns:
                    continue
                f.write(f"<h3>{m_label}</h3>")
                report = generate_phase_report(df_slice, name, override_response=m_col)
                render_report_body(f, report)
                f.write("<hr>")

            f.write("</div>")
            sub_first = False


def _write_footer(f: Any) -> None:
    f.write("</div></body></html>")


def render_report_body(f: TextIO, report: dict[str, Any]) -> None:
    """Helper to render the 5-layer components into the file stream."""
    if "error" in report:
        f.write(f"<p class='error'>Analysis skipped: {report['error']}</p>")
        return

    # Layer A: Structure
    f.write(f"""
    <div class="layer">
        <div class="layer-title"><span class="layer-badge">A</span> Structure Layer (ANOVA & Interactions)</div>
        <div class="grid">
            <div>{report["structure"][0].to_html(full_html=False, include_plotlyjs=False)}</div>
            <div>{report["structure"][1].to_html(full_html=False, include_plotlyjs=False)}</div>
        </div>
    </div>
    """)

    # Layer B: Influence
    f.write(f"""
    <div class="layer">
        <div class="layer-title"><span class="layer-badge">B</span> Influence Layer (Sensitivity & Stiffness)</div>
        <div class="grid">
            <div>{report["influence"][0].to_html(full_html=False, include_plotlyjs=False)}</div>
            <div>{report["influence"][1].to_html(full_html=False, include_plotlyjs=False)}</div>
        </div>
    </div>
    """)

    # Layer C: Shape
    f.write(f"""
    <div class="layer">
        <div class="layer-title"><span class="layer-badge">C</span> Shape Layer (Quality & Topology)</div>
        <div class="grid">
            <div class="full-width">{report["shape"][0].to_html(full_html=False, include_plotlyjs=False)}</div>
            <div>{report["shape"][1].to_html(full_html=False, include_plotlyjs=False)}</div>
            <div>{report["shape"][2].to_html(full_html=False, include_plotlyjs=False)}</div>
        </div>
    </div>
    """)

    # Layer E: Tables
    f.write(f"""
    <div class="layer">
        <div class="layer-title"><span class="layer-badge">E</span> Statistical Record</div>
        <div class="grid">
            <div>{report["tables"]["anova"]}</div>
        </div>
    </div>
    """)


if __name__ == "__main__":
    generate_dashboard()
