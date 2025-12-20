import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import splrep, splev

def get_common_layout(title: str, xaxis_title: str = "Crank Angle (deg)", yaxis_title: str = ""):
    """Return a standard consistent Plotly layout."""
    return go.Layout(
        title=title,
        xaxis=dict(title=xaxis_title, range=[0, 360]),
        yaxis=dict(title=yaxis_title),
        template="plotly_white",
        height=600
    )

def smooth_curve(theta_arr: np.ndarray, values: np.ndarray, s_factor: float = 0.2, floor: float = 0.15) -> np.ndarray:
    """
    Fit B-Spline to data to remove jitter (e.g. for Valve Profiles).
    
    Args:
        theta_arr: Crank angle array (deg/rad doesn't matter, usually deg).
        values: Data array.
        s_factor: Smoothing factor (higher = smoother). Default 0.2 (aggressive).
        floor: Noise floor. Values < floor are zeroed before fit.
    
    Returns:
        Smoothed array of same length.
    """
    vals_clean = np.array(values)
    
    # 1. Pre-filter noise
    vals_clean[vals_clean < floor] = 0.0
    
    # Check for empty signal
    if np.max(vals_clean) < floor:
        return np.zeros_like(theta_arr)

    try:
        # Fit B-Spline
        tck = splrep(theta_arr, vals_clean, s=s_factor)
        smoothed = splev(theta_arr, tck)
        
        # 2. Post-clip negatives
        smoothed[smoothed < 0.01] = 0.0
        
        return smoothed
    except Exception:
        # Fallback if spline fails (e.g. not enough points)
        return vals_clean
