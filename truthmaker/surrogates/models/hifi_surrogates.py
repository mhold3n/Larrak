"""Thermal Surrogate Model for T_crown_max prediction.

Replaces 1D thermal model with trained neural network
for fast feasibility checking in CEM.
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from truthmaker.surrogates.models.ensemble import BoundedMLP, EnsembleSurrogate


class ThermalSurrogate(EnsembleSurrogate):
    """
    Ensemble surrogate for piston crown temperature prediction.

    Inputs (5):
        - bore [normalized 0-1]
        - stroke [normalized 0-1]
        - compression_ratio [normalized 0-1]
        - rpm [normalized 0-1]
        - load [normalized 0-1]

    Outputs (1):
        - T_crown_max [K], bounded to [350, 700]

    Uncertainty:
        - Returns (mean, std) for ensemble disagreement
        - High std indicates extrapolation or sparse training data
    """

    # Physical bounds for output
    T_MIN = 350.0  # K
    T_MAX = 700.0  # K

    def __init__(
        self, n_models: int = 5, hidden_dims: list[int] | None = None, dropout_rate: float = 0.1
    ):
        """
        Initialize thermal surrogate ensemble.

        Args:
            n_models: Number of ensemble members
            hidden_dims: Hidden layer sizes, default [64, 64, 64]
            dropout_rate: Dropout for MC uncertainty
        """
        # Sigmoid output bounded to [0, 1], then scaled to [T_MIN, T_MAX]
        output_bounds = {
            0: ("sigmoid", 0.0, 1.0)  # Will scale externally
        }

        super().__init__(
            n_models=n_models,
            input_dim=5,
            output_dim=1,
            hidden_dims=hidden_dims or [64, 64, 64],
            output_bounds=output_bounds,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with temperature scaling.

        Returns:
            mean: Predicted T_crown_max in Kelvin
            std: Uncertainty in Kelvin
        """
        # Get normalized [0, 1] predictions
        mean_norm, std_norm = super().forward(x)

        # Scale to physical units
        temp_range = self.T_MAX - self.T_MIN
        mean_K = mean_norm * temp_range + self.T_MIN
        std_K = std_norm * temp_range

        return mean_K, std_K

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Numpy interface for prediction.

        Args:
            inputs: Array of shape (batch, 5) with normalized inputs

        Returns:
            (mean, std) in Kelvin
        """
        self.eval()
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32)
            mean, std = self.forward(x)
            return mean.numpy(), std.numpy()

    def is_feasible(
        self,
        inputs: np.ndarray,
        T_limit: float = 620.0,  # K
        confidence: float = 2.0,  # sigma
    ) -> tuple[bool, float, float]:
        """
        Check if predicted temperature is feasible.

        Args:
            inputs: Normalized input array (1, 5)
            T_limit: Maximum allowable temperature [K]
            confidence: Number of std deviations to add

        Returns:
            (is_feasible, T_predicted, uncertainty)
        """
        mean, std = self.predict(inputs)
        T_pred = float(mean[0, 0])
        T_std = float(std[0, 0])

        # Conservative: add uncertainty to prediction
        T_conservative = T_pred + confidence * T_std
        feasible = T_conservative < T_limit

        return feasible, T_pred, T_std


class FlowCoefficientSurrogate(EnsembleSurrogate):
    """
    Ensemble surrogate for discharge coefficient prediction.

    Inputs (5):
        - bore [normalized]
        - stroke [normalized]
        - cr [normalized]
        - rpm [normalized]
        - load [normalized]

    Outputs (1):
        - Cd_effective [], bounded to [0.3, 0.8]
    """

    CD_MIN = 0.3
    CD_MAX = 0.8

    def __init__(
        self, n_models: int = 5, hidden_dims: list[int] | None = None, dropout_rate: float = 0.1
    ):
        output_bounds = {0: ("sigmoid", 0.0, 1.0)}

        super().__init__(
            n_models=n_models,
            input_dim=5,
            output_dim=1,
            hidden_dims=hidden_dims or [64, 64, 64],
            output_bounds=output_bounds,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_norm, std_norm = super().forward(x)
        cd_range = self.CD_MAX - self.CD_MIN
        mean = mean_norm * cd_range + self.CD_MIN
        std = std_norm * cd_range
        return mean, std

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32)
            mean, std = self.forward(x)
            return mean.numpy(), std.numpy()


class StructuralSurrogate(EnsembleSurrogate):
    """
    Ensemble surrogate for von Mises stress prediction.

    Inputs (5):
        - bore, stroke, cr, rpm, load [normalized]

    Outputs (1):
        - von_mises_max [MPa], bounded to [0, 400]
    """

    STRESS_MIN = 0.0
    STRESS_MAX = 400.0

    def __init__(
        self, n_models: int = 5, hidden_dims: list[int] | None = None, dropout_rate: float = 0.1
    ):
        output_bounds = {0: ("sigmoid", 0.0, 1.0)}

        super().__init__(
            n_models=n_models,
            input_dim=5,
            output_dim=1,
            hidden_dims=hidden_dims or [64, 64, 64],
            output_bounds=output_bounds,
            dropout_rate=dropout_rate,
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mean_norm, std_norm = super().forward(x)
        stress_range = self.STRESS_MAX - self.STRESS_MIN
        mean = mean_norm * stress_range + self.STRESS_MIN
        std = std_norm * stress_range
        return mean, std

    def predict(self, inputs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.eval()
        with torch.no_grad():
            x = torch.tensor(inputs, dtype=torch.float32)
            mean, std = self.forward(x)
            return mean.numpy(), std.numpy()

    def is_feasible(
        self,
        inputs: np.ndarray,
        yield_strength: float = 280.0,  # MPa
        safety_factor: float = 1.5,
        confidence: float = 2.0,
    ) -> tuple[bool, float, float]:
        """
        Check if stress is below yield with safety factor.
        """
        mean, std = self.predict(inputs)
        stress_pred = float(mean[0, 0])
        stress_std = float(std[0, 0])

        # Conservative: add uncertainty
        stress_conservative = stress_pred + confidence * stress_std
        allowable = yield_strength / safety_factor
        feasible = stress_conservative < allowable

        return feasible, stress_pred, stress_std
