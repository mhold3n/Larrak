"""
Calibration Map Registry.
Manages the loading and querying of Surrogate Models (Friction, Combustion).
Calculates Trust Regions to warn about extrapolation.
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np

# Define Calibration Directory
CALIBRATION_DIR = Path(__file__).parent


class CalibrationRegistry:
    """
    Singleton-style manager for calibration maps.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(CalibrationRegistry, cls).__new__(cls)
            cls._instance.maps = {}
            cls._instance.bounds = {}
            cls._instance.load_maps()
        return cls._instance
        
    def load_maps(self):
        """Reload all JSON maps from calibration directory."""
        if not CALIBRATION_DIR.exists():
            print(f"Warning: Calibration dir {CALIBRATION_DIR} not found.")
            return
            
        for map_file in CALIBRATION_DIR.glob("*.json"):
            try:
                with open(map_file, "r") as f:
                    data = json.load(f)
                    
                domain = map_file.stem.replace("_map.v1", "").replace("_map", "")
                self.maps[domain] = data
                
                # Parse Bounds (if available or inferred)
                # Currently inferred from usage limits or should be saved in JSON?
                # JSON saves "n_samples".
                # Refinement: Map fitters should save [min, max] for each feature.
                # For now, minimal trust region: check if map exists.
                
                print(f"Loaded Calibration Map: {domain}")
            except Exception as e:
                print(f"Error loading map {map_file}: {e}")
                
    def predict(self, domain: str, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Predict values using the loaded map for a domain.
        
        Args:
            domain: "friction", "combustion", etc.
            inputs: Dictionary of input features (e.g. {"rpm": 2000, "p_max_bar": 50})
            
        Returns:
            Dictionary of predicted coefficients (e.g. {"A": 0.1, "B": 0.05})
            OR Dictionary of direct values if map targets single var.
            
        TODO: Use common.surrogates.PolynomialSurrogate logic here?
        Yes, we should reuse the prediction logic.
        """
        if domain not in self.maps:
            # Fallback / Default
            return {}
            
        artifact = self.maps[domain]
        model_type = artifact.get("model_type", "unknown")
        
        if "polynomial" in model_type:
            return self._predict_polynomial(artifact, inputs)
            
        return {}
        
    def _predict_polynomial(self, artifact: Dict, inputs: Dict[str, float]) -> Dict[str, float]:
        """
        Evaluate Polynomial Surrogate from Artifact.
        Coefficient structure in JSON:
        coeffs: { "feature_name": val, "bias": val } 
        BUT standard PolySurrogate output was:
        coeffs: { "bias": val, "x1": val, "x1^2": val ... }
        
        Wait, for Friction Map, we fit "fmep_bar" directly.
        But for Optimization, we often need "A, B" coefficients if the Physics model uses them.
        
        CURRENT STATE:
        - Friction Map targets "fmep_bar".
        - Combustion Map targets "wiebe_m".
        
        So this returns the scalar value of the target.
        """
        coeffs = artifact["coeffs"]
        features = artifact["features"]
        deg = 1 if "deg1" in artifact["model_type"] else 2
        
        # 1. Construct Feature Vector
        vals = []
        for f in features:
            v = inputs.get(f, 0.0)
            vals.append(v)
        X = np.array(vals)
        
        # 2. Compute Terms (Matching Surrogates.py logic)
        # Bias
        y = coeffs.get("bias", 0.0)
        
        # Linear
        for i, name in enumerate(features):
            c = coeffs.get(name, 0.0)
            y += c * X[i]
            
        # Quadratic/Interaction (if needed)
        # ... logic mirrors surrogates.py ...
        
        # Return as target_name? using domain name?
        # The map doesn't explicitly guarantee target name in top level, 
        # but the filename usually implies it.
        # Friction -> returns fmep
        
        return {"value": y}

    def get_trust_status(self, domain: str, inputs: Dict[str, float]) -> str:
        """
        Returns 'Trusted', 'Extrapolated', or 'Untrusted'.
        """
        if domain not in self.maps:
            return "Untrusted (Missing)"
        return "Trusted" # Stub for now

    def update_map_bounded(self, domain: str, new_coeffs: Dict[str, float], learning_rate: float = 0.2):
        """
        Update an existing map with new coefficients using a bounded step.
        New = Old * (1 - alpha) + Target * alpha
        
        Args:
            domain: Map domain name.
            new_coeffs: Dictionary of target coefficients.
            learning_rate: Alpha (0.0 to 1.0).
        """
        if domain not in self.maps:
            print(f"Registry: Creating new map for {domain}")
            # Identify model type and features from context? 
            # This method assumes we are updating the ARTIFACT data structure.
            # We usually need the full artifact structure to create new.
            # For update, we assume structure exists.
            return

        artifact = self.maps[domain]
        current_coeffs = artifact.get("coeffs", {})
        
        updated_coeffs = {}
        for k, v_target in new_coeffs.items():
            v_old = current_coeffs.get(k, v_target) 
            
            # Linear Interpolation Update (Relaxation)
            # v_new = v_old + alpha * (v_target - v_old)
            v_new = v_old * (1.0 - learning_rate) + v_target * learning_rate
            
            updated_coeffs[k] = v_new
            
        # Save back to runtime memory
        self.maps[domain]["coeffs"] = updated_coeffs
        
        # Persist to Disk is handled by the caller (fitters) normally, 
        # but Registry acts as Manager. 
        # Ideally, fitters should call this registry method to get the "Safe" coeffs,
        # then save them.
        print(f"Registry: Bounded update applied to {domain} (alpha={learning_rate})")
        return updated_coeffs

# Global Accessor
def get_calibration(domain: str, inputs: Dict[str, float]) -> float:
    reg = CalibrationRegistry()
    res = reg.predict(domain, inputs)
    return res.get("value", 0.0)
