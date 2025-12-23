import os

import pandas as pd
import plotly.graph_objects as go


def plot_efficiency_map(df: pd.DataFrame, x_col: str, y_col: str, z_col: str, output_path: str = None) -> go.Figure:
    """
    Generate 3D Scatter/Mesh map for Efficiency (DOE results).
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=df[x_col],
        y=df[y_col],
        z=df[z_col],
        mode='markers',
        marker=dict(
            size=5,
            color=df[z_col],
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title=z_col)
        )
    )])
    
    fig.update_layout(
        title=f"Efficiency Map ({z_col} vs {x_col}/{y_col})",
        scene=dict(
            xaxis_title=x_col,
            yaxis_title=y_col,
            zaxis_title=z_col,
        ),
        template="plotly_white",
        height=800
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"Saved efficiency map to {output_path}")
        
    return fig

def plot_efficiency_surface(df: pd.DataFrame, output_path: str = None) -> go.Figure:
    """
    Generate 3D Surface/Scatter Map:
    X: RPM
    Y: Air Load (P_int)
    Z: Fuel Load
    Color: Efficiency
    """
    x = df["rpm"]
    y = df["p_int_bar"] # Proxy for Air Mass
    z = df["fuel_mass_mg"]
    c = df["thermal_efficiency"]
    
    fig = go.Figure()
    
    # 1. Scatter Points (Truth Data)
    fig.add_trace(go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=5,
            color=c,
            colorscale='Viridis',
            opacity=0.8,
            colorbar=dict(title="Efficiency")
        ),
        name="DOE Points"
    ))
    
    # 2. Mesh Surface (Interpolated Visual)
    # Allows seeing the 'shape' of the map
    fig.add_trace(go.Mesh3d(
        x=x, y=y, z=z,
        intensity=c,
        colorscale='Viridis',
        opacity=0.2, # Transparent surface to see points
        name="Surface"
    ))

    fig.update_layout(
        title="Efficiency Map Surface (X=RPM, Y=Air/Boost, Z=Fuel)",
        scene=dict(
            xaxis_title="RPM",
            yaxis_title="Air Load (P_int bar)",
            zaxis_title="Fuel Mass (mg)",
        ),
        template="plotly_white",
        height=800
    )
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"Saved specific efficiency surface to {output_path}")
        
    return fig
