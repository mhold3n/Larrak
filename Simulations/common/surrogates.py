"""
Surrogate Modeling Utilities.
Used to fit calibration maps (Physics -> Simplified Coefficients).
"""

import numpy as np
from typing import List, Dict, Any, Optional

class PolynomialSurrogate:
    """
    Multivariate Polynomial/Linear Surrogate.
    y = c0 + c1*x1 + c2*x2 + ... + Interaction Terms
    """
    
    def __init__(self, feature_names: List[str], target_name: str, degree: int = 1, interaction: bool = False):
        self.feature_names = feature_names
        self.target_name = target_name
        self.degree = degree
        self.interaction = interaction
        self.coeffs: Optional[np.ndarray] = None
        self.intercept: float = 0.0
        self.feature_map_names: List[str] = []
        
        if degree > 2:
            raise NotImplementedError("Degrees > 2 not supported yet (Combinatorial explosion)")

    def _create_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """Expand X into [1, x1, x2, x1^2, x1x2, ...]"""
        n_samples, n_features = X.shape
        
        # 1. Bias handled by lstsq or explicit column? 
        # Explicit column [1] makes extracting intercept easy.
        
        features = [np.ones((n_samples, 1))]
        names = ["bias"]
        
        # Linear
        features.append(X)
        names.extend(self.feature_names)
        
        # Quadratic / Interaction
        if self.degree == 2:
            # Squared
            features.append(X**2)
            names.extend([f"{n}^2" for n in self.feature_names])
            
            if self.interaction:
                # Pairwise
                for i in range(n_features):
                    for j in range(i+1, n_features):
                        col = (X[:, i] * X[:, j]).reshape(-1, 1)
                        features.append(col)
                        names.append(f"{self.feature_names[i]}*{self.feature_names[j]}")
                        
        self.feature_map_names = names
        return np.hstack(features)

    def fit(self, data: List[Dict[str, float]]):
        """Fit coefficients to list of simulation results."""
        print(f"DEBUG: Starting fit with {len(data)} records.")
        if not data:
            raise ValueError("No data provided for fitting")
            
        # Extract X and y
        X_list = []
        y_list = []
        
        for record in data:
            row = [record.get(f, 0.0) for f in self.feature_names]
            X_list.append(row)
            y_list.append(record.get(self.target_name, 0.0))
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        # Design Matrix
        A = self._create_design_matrix(X)
        
        # Solve Least Squares
        # A * c = y
        # Using Ridge Regression for stability and to avoid potential MKL/lstsq crashes
        # (A.T @ A + alpha * I) c = A.T @ y
        
        print(f"DEBUG: Solving with Ridge Regression (alpha=1e-6)")
        alpha = 1e-6
        ATA = A.T @ A
        ATy = A.T @ y
        
        # Regularize diagonal
        reg = alpha * np.eye(ATA.shape[0])
        # Don't regularize bias (index 0) as strongly? Or usually ok.
        
        try:
            c = np.linalg.solve(ATA + reg, ATy)
            residuals = y - A @ c # Manual residuals
            rank = A.shape[1]
        except Exception as e:
            print(f"DEBUG: Linear Solve Failed: {e}")
            raise
            
        print("DEBUG: Solve Complete")
        
        self.coeffs = c
        self.intercept = c[0]
        
        # Convert to Python floats for JSON serialization
        coeffs_dict = {name: float(val) for name, val in zip(self.feature_map_names, c)}
        
        return {
            "coeffs": coeffs_dict,
            "residuals": float(residuals[0]) if len(residuals) > 0 else 0.0,
            "r2": float(self._calc_r2(y, A @ c))
        }
        
    def predict(self, input_dict: Dict[str, float]) -> float:
        if self.coeffs is None:
            raise RuntimeError("Model not fitted")
        
        row = [input_dict.get(f, 0.0) for f in self.feature_names]
        X = np.array([row])
        A = self._create_design_matrix(X)
        return float(A @ self.coeffs)
        
    def _calc_r2(self, y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1.0 - (ss_res / (ss_tot + 1e-8))


class WiebeParameterSurrogate:
    """
    Predicts Wiebe Combustion Parameters (a, m, start, duration)
    based on operating conditions (RPM, Load, etc.) using
    underlying PolynomialSurrogates for each parameter.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        # We model 4 parameters usually: a, m, start_angle, duration
        self.models = {
            "a": PolynomialSurrogate(feature_names, "a", degree=2, interaction=True),
            "m": PolynomialSurrogate(feature_names, "m", degree=2, interaction=True),
            "start": PolynomialSurrogate(feature_names, "start", degree=2, interaction=True),
            "duration": PolynomialSurrogate(feature_names, "duration", degree=2, interaction=True)
        }
        
    def fit(self, data: List[Dict[str, float]]):
        """
        Data must contain keys: 'a', 'm', 'start', 'duration' 
        plus the feature_names (e.g. 'rpm', 'load').
        """
        results = {}
        for param, model in self.models.items():
            print(f"Fitting Wiebe Parameter: {param}")
            try:
                fit_res = model.fit(data)
                results[param] = fit_res
            except Exception as e:
                print(f"Error fitting {param}: {e}")
                results[param] = {"success": False}
        return results

    def predict(self, input_dict: Dict[str, float]) -> Dict[str, float]:
        return {
            param: model.predict(input_dict)
            for param, model in self.models.items()
        }
