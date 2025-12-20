"""
Delta Learner: Multi-fidelity correction learning.

Learns y_high ≈ y_low + Δ(x) from sparse high-fidelity labels.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn

from .ensemble import BoundedMLP


class DeltaLearner(nn.Module):
    """
    Multi-fidelity learning: predicts high-fidelity from low-fidelity + correction.
    
    y_high ≈ y_low + Δ(x)
    
    The delta network learns the residual between low and high fidelity,
    which is typically easier to learn than the full y_high directly.
    
    Usage:
        delta = DeltaLearner(input_dim=3, output_dim=3)
        y_high_pred = delta(x, y_low)
        
        # Training: 
        loss = criterion(y_high_pred, y_high_actual)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 3,
        hidden_dims: List[int] = None,
        include_y_low_as_input: bool = True,
        dropout_rate: float = 0.0
    ):
        """
        Args:
            input_dim: Number of input features (design variables)
            output_dim: Number of outputs to correct
            hidden_dims: Hidden layer sizes for delta network
            include_y_low_as_input: If True, delta_net takes [x, y_low] as input
            dropout_rate: Dropout for regularization
        """
        super().__init__()
        
        self.include_y_low = include_y_low_as_input
        self.output_dim = output_dim
        
        if hidden_dims is None:
            hidden_dims = [32, 32]  # Smaller network for delta (residuals are simpler)
        
        # Delta network input dimension
        delta_input_dim = input_dim + output_dim if include_y_low_as_input else input_dim
        
        self.delta_net = BoundedMLP(
            input_dim=delta_input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            output_bounds=None,  # Delta has no bounds (can be positive or negative)
            dropout_rate=dropout_rate
        )
    
    def forward(
        self, 
        x: torch.Tensor, 
        y_low: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict high-fidelity output.
        
        Args:
            x: Design inputs of shape (batch, input_dim)
            y_low: Low-fidelity predictions of shape (batch, output_dim)
            
        Returns:
            y_high_pred: Predicted high-fidelity of shape (batch, output_dim)
        """
        if self.include_y_low:
            delta_input = torch.cat([x, y_low], dim=-1)
        else:
            delta_input = x
        
        delta = self.delta_net(delta_input)
        return y_low + delta
    
    def predict_delta(
        self, 
        x: torch.Tensor, 
        y_low: torch.Tensor
    ) -> torch.Tensor:
        """Return only the delta (correction) without adding to y_low."""
        if self.include_y_low:
            delta_input = torch.cat([x, y_low], dim=-1)
        else:
            delta_input = x
        return self.delta_net(delta_input)


class MultiFidelitySurrogate(nn.Module):
    """
    Complete multi-fidelity pipeline: low-fidelity model + delta correction.
    
    Combines:
    1. Base surrogate (trained on abundant low-fidelity data)
    2. Delta learner (trained on sparse high-fidelity data)
    
    Usage:
        mf_model = MultiFidelitySurrogate(base_model, delta_model)
        
        # Low-fidelity prediction only
        y_low = mf_model.predict_low(x)
        
        # High-fidelity prediction (low + delta)
        y_high = mf_model(x)
    """
    
    def __init__(
        self,
        base_model: nn.Module,
        delta_model: Optional[DeltaLearner] = None
    ):
        """
        Args:
            base_model: Low-fidelity surrogate (e.g., EnsembleSurrogate)
            delta_model: Delta correction network (optional, can add later)
        """
        super().__init__()
        self.base_model = base_model
        self.delta_model = delta_model
    
    def predict_low(self, x: torch.Tensor) -> torch.Tensor:
        """Predict using low-fidelity model only."""
        out = self.base_model(x)
        # Handle ensemble output (mean, std) vs regular output
        if isinstance(out, tuple):
            return out[0]  # Return mean
        return out
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict high-fidelity (low + delta correction)."""
        y_low = self.predict_low(x)
        
        if self.delta_model is None:
            return y_low
        
        return self.delta_model(x, y_low)
    
    def add_delta_model(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int] = None
    ) -> None:
        """Add delta model after training base on low-fidelity data."""
        self.delta_model = DeltaLearner(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims
        )
    
    def freeze_base(self) -> None:
        """Freeze base model weights for delta-only training."""
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def unfreeze_base(self) -> None:
        """Unfreeze base model for joint fine-tuning."""
        for param in self.base_model.parameters():
            param.requires_grad = True
