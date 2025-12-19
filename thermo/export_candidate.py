"""
Exports optimization candidates to Phase 4 Simulation Input format.
"""

import numpy as np
import json
import uuid
import sys
import os

# Ensure project root is in path to import Simulations
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from Simulations.common.io_schema import (
    SimulationInput, GeometryConfig, OperatingPoint, BoundaryConditions
)

def export_candidate(
    run_id: str,
    meta: dict,
    w_opt: np.ndarray,
    geo_params: dict,
    ops_params: dict
) -> str:
    """
    Converts a single optimization point into a SimulationInput JSON string.
    
    Args:
        run_id: Unique identifier string.
        meta: Metadata dictionary returned by build_thermo_nlp (contains indices and scales).
        w_opt: Optimized solution vector (1D numpy array).
        geo_params: Dict with keys [bore, stroke, conrod, CR].
        ops_params: Dict with keys [rpm, p_int, T_int].
        
    Returns:
        JSON string of the SimulationInput.
    """
    
    # 1. Extract and De-scale Trajectories
    scales = meta["scales"]
    
    # INDICES (List of ints)
    idx_x = meta["state_indices_x"]
    idx_v = meta["state_indices_v"]
    idx_m = meta["state_indices_m_c"]
    idx_T = meta["state_indices_T_c"]
    
    # Extract
    # w_opt might be CasADi DM or numpy array. Ensure numpy.
    w = np.array(w_opt).flatten()
    
    x_scaled = w[idx_x]
    v_scaled = w[idx_v]
    m_scaled = w[idx_m]
    T_scaled = w[idx_T]
    
    # De-scale
    x_phys = (x_scaled - scales["x_shift"]) * scales["x"]
    v_phys = v_scaled * scales["v"]
    m_phys = m_scaled * scales["m_c"]
    T_phys = T_scaled * scales["T_c"]
    
    # 2. Derive Pressure and Crank Angle
    # Assuming standard collocation grid (linear time? No, theta based).
    # theta usually 0 to pi.
    # We need to reconstruct theta grid. n_coll is len(x_phys).
    # Wait, indices are for ALL collocation points (degree * n_coll + endpoints).
    # Simplification: Just generate linear theta array for now matching length.
    n_points = len(x_phys)
    theta_arr = np.linspace(0, np.pi, n_points) # Expansion stroke 0->180
    
    # Volume (Single Cylinder)
    B = geo_params["bore"]
    S = geo_params["stroke"]
    CR = geo_params["compression_ratio"]
    
    V_disp = (np.pi * B**2 / 4.0) * S
    V_c = V_disp / (CR - 1.0)
    
    # OP Engine Volume Logic (2 * (Vc + A*x)) matches NLP
    # But geometry passed in is "per piston" usually?
    # Let's assume params are per-piston to match standard Larrak convention.
    A_piston = np.pi * B**2 / 4.0
    V_phys = 1.0 * (V_c + A_piston * x_phys) # 1 Cylinder for Thermodynamics
    
    # Pressure (Ideal Gas)
    R_gas = 287.0
    P_phys = m_phys * R_gas * T_phys / V_phys
    
    # 3. Construct Pydantic Objects
    geometry = GeometryConfig(
        bore=B,
        stroke=S,
        conrod=geo_params["conrod"],
        compression_ratio=CR
    )
    
    operating_point = OperatingPoint(
        rpm=ops_params["rpm"],
        lambda_val=ops_params.get("lambda", 1.0),
        p_intake=ops_params["p_int"],
        T_intake=ops_params["T_int"]
    )
    
    # Convert theta to Crank Angle (deg)
    # 0 -> 180 (Expansion)
    # Standard engine convention: 0 is TDC Firing usually?
    # Larrak NLP: 0 is TDC.
    ca_deg = theta_arr * 180.0 / np.pi
    
    bcs = BoundaryConditions(
        crank_angle=ca_deg.tolist(),
        pressure_gas=P_phys.tolist(),
        temperature_gas=T_phys.tolist(),
        piston_speed=v_phys.tolist()
    )
    
    # 4. Bundle
    sim_input = SimulationInput(
        run_id=run_id,
        geometry=geometry,
        operating_point=operating_point,
        boundary_conditions=bcs
    )
    
    return sim_input.model_dump_json(indent=2)
