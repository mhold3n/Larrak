"""
Ensemble Surrogate: Multiple MLPs with uncertainty quantification.

Provides epistemic uncertainty via ensemble disagreement.
"""

from __future__ import annotations

from typing import Tuple, Optional, Dict, List
import torch
import torch.nn as nn
import numpy as np


class BoundedMLP(nn.Module):
    """
    MLP with optional bounded output layers.
    
    Supports:
    - sigmoid: output in [0, 1] (scaled to [lo, hi])
    - softplus: output in [0, inf)
    - none: unconstrained
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        output_dim: int = 3,
        hidden_dims: List[int] = None,
        output_bounds: Optional[Dict[int, Tuple[str, float, float]]] = None,
        dropout_rate: float = 0.0
    ):
        """
        Args:
            input_dim: Number of input features
            output_dim: Number of outputs
            hidden_dims: List of hidden layer sizes, default [64, 64, 64]
            output_bounds: Dict mapping output index to (transform, lo, hi)
                e.g., {0: ('sigmoid', 0, 1), 1: ('softplus', 0, None)}
            dropout_rate: Dropout rate for MC dropout uncertainty
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [64, 64, 64]
        
        self.output_bounds = output_bounds or {}
        self.dropout_rate = dropout_rate
        
        # Build layers
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional output transforms."""
        raw_output = self.net(x)
        
        if not self.output_bounds:
            return raw_output
        
        # Apply transforms to each bounded output
        outputs = []
        for i in range(raw_output.shape[-1]):
            if i in self.output_bounds:
                transform, lo, hi = self.output_bounds[i]
                col = raw_output[..., i:i+1]
                
                if transform == 'sigmoid':
                    # Sigmoid maps to [0,1], then scale to [lo, hi]
                    col = torch.sigmoid(col) * (hi - lo) + lo
                elif transform == 'softplus':
                    # Softplus maps to [0, inf), offset by lo
                    col = nn.functional.softplus(col) + (lo if lo else 0)
                # 'none' or unknown: keep raw
                outputs.append(col)
            else:
                outputs.append(raw_output[..., i:i+1])
        
        return torch.cat(outputs, dim=-1)
    
    def enable_mc_dropout(self) -> None:
        """Enable dropout during inference for MC dropout uncertainty."""
        for module in self.modules():
            if isinstance(module, nn.Dropout):
                module.train()


class EnsembleSurrogate(nn.Module):
    """
    Ensemble of MLPs with uncertainty quantification.
    
    Computes epistemic uncertainty via ensemble disagreement (variance across models).
    Supports both ensemble and MC dropout uncertainty.
    
    Usage:
        model = EnsembleSurrogate(n_models=5, input_dim=3, output_dim=3)
        mean, std = model(x)  # Returns mean prediction and uncertainty
        
        # For training individual models:
        for i, m in enumerate(model.models):
            loss = criterion(m(x), y)
    """
    
    def __init__(
        self,
        n_models: int = 5,
        input_dim: int = 3,
        output_dim: int = 3,
        hidden_dims: List[int] = None,
        output_bounds: Optional[Dict[int, Tuple[str, float, float]]] = None,
        dropout_rate: float = 0.0
    ):
        """
        Args:
            n_models: Number of ensemble members
            input_dim: Number of input features
            output_dim: Number of outputs
            hidden_dims: List of hidden layer sizes per model
            output_bounds: Output transform bounds (shared across ensemble)
            dropout_rate: Dropout rate for MC dropout
        """
        super().__init__()
        
        self.n_models = n_models
        self.output_dim = output_dim
        
        self.models = nn.ModuleList([
            BoundedMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                output_bounds=output_bounds,
                dropout_rate=dropout_rate
            )
            for _ in range(n_models)
        ])
        
        # Calibrated uncertainty threshold (set via calibrate())
        self._uncertainty_threshold: Optional[torch.Tensor] = None
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning mean prediction and epistemic uncertainty.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            mean: Mean prediction of shape (batch, output_dim)
            std: Standard deviation (uncertainty) of shape (batch, output_dim)
        """
        # Stack predictions from all models: (n_models, batch, output_dim)
        preds = torch.stack([m(x) for m in self.models], dim=0)
        
        mean = preds.mean(dim=0)
        std = preds.std(dim=0)
        
        return mean, std
    
    def predict_with_samples(self, x: torch.Tensor) -> torch.Tensor:
        """Return all ensemble member predictions."""
        return torch.stack([m(x) for m in self.models], dim=0)
    
    def reject_threshold(
        self, 
        uncertainty: torch.Tensor,
        threshold: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Binary mask: True where uncertainty exceeds threshold.
        
        Args:
            uncertainty: Uncertainty tensor from forward()
            threshold: Override threshold, or use calibrated value
            
        Returns:
            Boolean tensor where True = reject (high uncertainty)
        """
        thresh = threshold if threshold is not None else self._uncertainty_threshold
        if thresh is None:
            # Default: reject if uncertainty > 2x mean uncertainty seen during calibration
            thresh = uncertainty.mean() * 2
        
        # Reject if ANY output dimension exceeds threshold
        return (uncertainty > thresh).any(dim=-1)
    
    def calibrate(
        self, 
        val_loader: 'torch.utils.data.DataLoader',
        coverage: float = 0.95
    ) -> None:
        """
        Calibrate uncertainty threshold on validation data.
        
        Sets threshold such that `coverage` fraction of predictions have
        uncertainty below threshold.
        
        Args:
            val_loader: Validation data loader
            coverage: Target coverage (e.g., 0.95 for 95% confidence)
        """
        self.eval()
        all_uncertainties = []
        
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0] if isinstance(batch, (list, tuple)) else batch
                _, std = self(x)
                all_uncertainties.append(std)
        
        all_std = torch.cat(all_uncertainties, dim=0)
        # Take max uncertainty across output dimensions per sample
        max_std = all_std.max(dim=-1).values
        
        # Find threshold for desired coverage
        self._uncertainty_threshold = torch.quantile(max_std, coverage)
    
    def get_model(self, idx: int) -> BoundedMLP:
        """Get individual ensemble member for training."""
        return self.models[idx]
    
    def save(self, path: str) -> None:
        """Save ensemble state dict."""
        torch.save({
            'state_dict': self.state_dict(),
            'n_models': self.n_models,
            'output_dim': self.output_dim,
            'threshold': self._uncertainty_threshold.item() if self._uncertainty_threshold else None
        }, path)
    
    @classmethod
    def load(cls, path: str, **kwargs) -> 'EnsembleSurrogate':
        """Load ensemble from saved state."""
        checkpoint = torch.load(path, weights_only=True)
        model = cls(n_models=checkpoint['n_models'], **kwargs)
        model.load_state_dict(checkpoint['state_dict'])
        if checkpoint.get('threshold'):
            model._uncertainty_threshold = torch.tensor(checkpoint['threshold'])
        return model
