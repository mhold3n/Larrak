"""
Curvature computation component.

This module provides a modular component for computing curvature
and osculating radius of cam curves.
"""

from typing import Dict, Any, List
import numpy as np

from ..base import BaseComponent, ComponentResult, ComponentStatus
from campro.logging import get_logger

log = get_logger(__name__)


class CurvatureComponent(BaseComponent):
    """
    Component for computing curvature and osculating radius of cam curves.
    
    This component computes the curvature κ(θ) and osculating radius ρ(θ)
    for polar curves defined by r(θ).
    """
    
    def _validate_parameters(self) -> None:
        """Validate component parameters."""
        # No specific parameters required for curvature computation
        pass
    
    def compute(self, inputs: Dict[str, np.ndarray]) -> ComponentResult:
        """
        Compute curvature and osculating radius.
        
        Parameters
        ----------
        inputs : Dict[str, np.ndarray]
            Input data containing:
            - 'theta': Cam angles (radians)
            - 'r_theta': Radius values r(θ)
            
        Returns
        -------
        ComponentResult
            Result containing curvature and osculating radius
        """
        try:
            # Validate inputs
            if not self.validate_inputs(inputs):
                return ComponentResult(
                    status=ComponentStatus.FAILED,
                    outputs={},
                    metadata={},
                    error_message="Invalid inputs"
                )
            
            theta = inputs['theta']
            r_theta = inputs['r_theta']
            
            log.info(f"Computing curvature for {len(theta)} points")
            
            # Compute derivatives using finite differences
            r_prime = self._compute_derivative(theta, r_theta)
            r_double_prime = self._compute_derivative(theta, r_prime)
            
            # Compute curvature for polar curve: κ = (r² + 2r'² - rr'') / (r² + r'²)^(3/2)
            r_squared = r_theta**2
            r_prime_squared = r_prime**2
            
            numerator = r_squared + 2 * r_prime_squared - r_theta * r_double_prime
            denominator = (r_squared + r_prime_squared)**(3/2)
            
            # Avoid division by zero
            kappa = np.divide(numerator, denominator, 
                            out=np.zeros_like(numerator), 
                            where=denominator != 0)
            
            # Compute osculating radius: ρ = 1/κ
            rho = np.divide(1.0, kappa, 
                          out=np.full_like(kappa, np.inf), 
                          where=kappa != 0)
            
            # Prepare outputs
            outputs = {
                'kappa': kappa,
                'rho': rho,
                'r_prime': r_prime,
                'r_double_prime': r_double_prime
            }
            
            # Prepare metadata
            metadata = {
                'num_points': len(theta),
                'min_curvature': float(np.min(kappa[np.isfinite(kappa)])),
                'max_curvature': float(np.max(kappa[np.isfinite(kappa)])),
                'min_osculating_radius': float(np.min(rho[np.isfinite(rho)])),
                'max_osculating_radius': float(np.max(rho[np.isfinite(rho)]))
            }
            
            log.info(f"Curvature computed successfully: kappa range [{metadata['min_curvature']:.3f}, {metadata['max_curvature']:.3f}]")
            
            return ComponentResult(
                status=ComponentStatus.COMPLETED,
                outputs=outputs,
                metadata=metadata
            )
            
        except Exception as e:
            log.error(f"Error computing curvature: {e}")
            return ComponentResult(
                status=ComponentStatus.FAILED,
                outputs={},
                metadata={},
                error_message=str(e)
            )
    
    def _compute_derivative(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute derivative using finite differences.
        
        Parameters
        ----------
        x : np.ndarray
            Independent variable
        y : np.ndarray
            Dependent variable
            
        Returns
        -------
        np.ndarray
            Derivative dy/dx
        """
        if len(x) < 2:
            return np.zeros_like(y)
        
        # Use central differences for interior points
        dy = np.gradient(y, x)
        
        return dy
    
    def get_required_inputs(self) -> List[str]:
        """Get list of required input names."""
        return ['theta', 'r_theta']
    
    def get_outputs(self) -> List[str]:
        """Get list of output names."""
        return ['kappa', 'rho', 'r_prime', 'r_double_prime']
