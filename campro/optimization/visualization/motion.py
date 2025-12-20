import plotly.graph_objects as go
import numpy as np
import os
from .utils import get_common_layout

def plot_motion_family(results: list, title: str, output_path: str = None) -> go.Figure:
    """
    Generate plot for Motion Law Family.
    Args:
        results: List of dicts with 'theta' (rad or deg), 'x_opt', 'label'/'fuel_mg'.
    """
    fig = go.Figure()
    
    for res in results:
        # Handle theta units (assume radians if max < 10?)
        theta = np.array(res["theta"])
        if np.max(theta) < 7.0: # 2pi ~ 6.28
            theta = np.degrees(theta)
            
        label = res.get("label", f"{res.get('fuel_mg', '?')} mg")
        
        fig.add_trace(go.Scatter(
            x=theta, 
            y=res["x_opt"], 
            mode='lines',
            name=label
        ))
        
    layout = get_common_layout(title, yaxis_title="Piston Position (m)")
    fig.update_layout(layout)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"Saved motion family plot to {output_path}")
        
    return fig
