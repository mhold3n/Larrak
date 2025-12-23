"""
Main Interface for Phase 2 Interpreter.
"""

import numpy as np
from .inversion import calculate_ideal_ratio
from .fitting import RatioCurveFitter


class Interpreter:
    """
    Facade for the Phase 2 Interpretation Process.
    """

    def __init__(self):
        pass

    def process(self, motion_data: dict, geometry_params: dict, options: dict = None) -> dict:
        """
        Process Phase 1 motion data into Phase 3 gear ratio profile.

        Args:
            motion_data: Dict containing 'x', 'v', 'theta' (or time).
                         Vectors should cover one full cycle.
            geometry_params: Dict with 'stroke', 'conrod'.
            options: Dict with 'mean_ratio', 'weights', 'n_knots', 'degree'.

        Returns:
            Dict containing the optimized ratio profile and metadata.
        """
        options = options or {}

        # 1. Extract Data
        # Expecting numpy arrays or lists
        x = np.array(motion_data["x"])
        v = np.array(motion_data["v"])
        theta = np.array(motion_data.get("theta", []))

        if len(theta) == 0 and "time" in motion_data:
            theta = np.array(motion_data["time"])  # Assuming time maps to angle linearly

        # 2. Extract Geometry
        stroke = float(geometry_params["stroke"])
        conrod = float(geometry_params["conrod"])

        # 3. Kinematic Inversion -> Ideal Ratio
        # Assuming theta is the Ring Angle phi? No, theta is Cycle Angle.
        # If we assume constant ring speed, phi = k * theta.
        # Usually Phase 1 normalizes cycle to 2pi or similar.
        # Phase 3 expects phi to be Ring Angle.
        # Let's assume 1-to-1 mapping for the base grid, or re-grid to uniform phi?

        # We need uniform grid for fitting? Fitting accepts any grid if basis evaluated there.
        # Let's map theta directly to phi_grid.
        phi_grid = theta

        i_ideal = calculate_ideal_ratio(x, v, stroke, conrod)

        # 4. Parametric Fitting
        mean_ratio = options.get("mean_ratio", 2.0)
        n_knots = options.get("n_knots", 50)
        degree = options.get("degree", 3)
        weights = options.get("weights", {"track": 1.0, "smooth": 0.1})

        fitter = RatioCurveFitter(n_knots=n_knots, degree=degree)

        result = fitter.fit(
            phi_grid=phi_grid, i_ideal=i_ideal, mean_ratio_target=mean_ratio, weights=weights
        )

        # 5. Format Output
        output = {
            "interpreter_version": "1.0",
            "status": result["status"],
            "geometry": geometry_params,
            "profile": {
                "phi": result["phi_grid"],
                "ratio": result["i_fitted"],
                "ratio_ideal": i_ideal.tolist(),
                "knots": result["knots"],
                "weights": result["weights"],
            },
            "meta": {
                "mean_ratio": result["mean_ratio"],
                "fitting_error": float(np.mean((np.array(result["i_fitted"]) - i_ideal) ** 2)),
            },
        }

        return output
