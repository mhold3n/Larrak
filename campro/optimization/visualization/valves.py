import os

import plotly.graph_objects as go

from .utils import get_common_layout, smooth_curve


def plot_valve_strategy(results: list, output_path: str = None) -> go.Figure:
    """
    Generate plot for Valve Strategy Load Sweep.
    Args:
        results: List of result dicts containing 'theta_x', 'x', 'theta_u', 'A_int', 'A_exh', 'fuel'.
        output_path: Optional path to save HTML.
    Returns:
        Plotly Figure.
    """
    fig = go.Figure()
    colors = ["blue", "green", "orange", "red", "purple", "cyan"]
    
    for i, res in enumerate(results):
        fuel = res.get("fuel", "Unknown")
        c = colors[i % len(colors)]
        
        # Piston Position (Solid)
        fig.add_trace(go.Scatter(
            x=res["theta_x"], y=res["x"],
            mode='lines', name=f'{fuel}mg Pos',
            line=dict(color=c, width=2)
        ))
        
        # Smooth Valves
        # Use util with consistent settings
        a_int_smooth = smooth_curve(res["theta_u"], res["A_int"], s_factor=0.2, floor=0.15)
        a_exh_smooth = smooth_curve(res["theta_u"], res["A_exh"], s_factor=0.2, floor=0.15)
        
        # Intake Area (Smoothed, Dash)
        fig.add_trace(go.Scatter(
            x=res["theta_u"], y=a_int_smooth,
            mode='lines', name=f'{fuel}mg Int',
            line=dict(color=c, dash='dash'),
            yaxis='y2'
        ))
        
        # Exhaust Area (Smoothed, Dot)
        fig.add_trace(go.Scatter(
            x=res["theta_u"], y=a_exh_smooth,
            mode='lines', name=f'{fuel}mg Exh',
            line=dict(color=c, dash='dot'),
            yaxis='y2'
        ))

    layout = get_common_layout(
        title="Valve Strategy Sweep (Variable Area Optimization)",
        yaxis_title="Piston Position (m)"
    )
    layout.yaxis2 = dict(
        title="Valve Open Fraction (Alpha)", 
        overlaying='y', 
        side='right', 
        range=[0, 1.1]
    )
    fig.update_layout(layout)
    
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.write_html(output_path)
        print(f"Saved valve strategy plot to {output_path}")
        
    return fig
