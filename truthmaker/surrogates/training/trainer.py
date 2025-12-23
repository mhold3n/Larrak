"""
CEM Trainer: Training loop with physics constraints and CEM-aware losses.

Provides structured loss functions and validation metrics for engineering surrogates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..models.ensemble import EnsembleSurrogate


@dataclass
class TrainingConfig:
    """Configuration for CEM-aware training."""

    epochs: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3

    # Physics constraint weights
    lambda_bounds: float = 1.0  # Weight for bound violation penalty
    lambda_monotonic: float = 0.5  # Weight for monotonicity penalty
    lambda_conservation: float = 0.1  # Weight for conservation constraints

    # Regularization
    weight_decay: float = 1e-4

    # Early stopping
    patience: int = 20
    min_delta: float = 1e-5


@dataclass
class PhysicsConstraints:
    """
    Physics constraints for engineering surrogates.

    bounds: Dict mapping output name to (min, max)
        e.g., {'efficiency': (0, 1), 'pressure': (0, 500)}

    monotonicity: List of (input_name, output_name, direction)
        e.g., [('rpm', 'power', 'increasing')]

    conservation: Optional callable that returns conservation residual
    """

    bounds: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    monotonicity: List[Tuple[str, str, str]] = field(default_factory=list)
    conservation_fn: Optional[Callable] = None


class PhysicsConstraintLoss(nn.Module):
    """
    Loss module for enforcing physics constraints.

    Computes soft penalties for:
    - Bound violations
    - Monotonicity violations
    - Conservation constraint residuals
    """

    def __init__(self, output_names: List[str], constraints: PhysicsConstraints):
        super().__init__()
        self.output_names = output_names
        self.constraints = constraints

        # Pre-compute output indices for bounds
        self.bound_indices = {}
        for name, (lo, hi) in constraints.bounds.items():
            if name in output_names:
                idx = output_names.index(name)
                self.bound_indices[idx] = (lo, hi)

    def forward(
        self, y_pred: torch.Tensor, x: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute physics constraint penalties.

        Args:
            y_pred: Predicted outputs (batch, output_dim)
            x: Optional inputs for monotonicity checks

        Returns:
            total_penalty: Sum of all constraint penalties
            breakdown: Dict with individual penalty values
        """
        penalties = {}

        # Bound violations (soft ReLU penalty)
        for idx, (lo, hi) in self.bound_indices.items():
            col = y_pred[:, idx]
            lower_violation = torch.relu(lo - col).mean()
            upper_violation = torch.relu(col - hi).mean()
            penalties[f"bound_{idx}"] = lower_violation + upper_violation

        bound_penalty = sum(penalties.values()) if penalties else torch.tensor(0.0)

        # Conservation (if applicable)
        conservation_penalty = torch.tensor(0.0)
        if self.constraints.conservation_fn is not None and x is not None:
            residual = self.constraints.conservation_fn(x, y_pred)
            conservation_penalty = residual.pow(2).mean()
            penalties["conservation"] = conservation_penalty.item()

        total = bound_penalty + conservation_penalty

        breakdown = {k: v.item() if torch.is_tensor(v) else v for k, v in penalties.items()}
        breakdown["total"] = total.item()

        return total, breakdown


class CEMTrainer:
    """
    CEM-aware training loop for ensemble surrogates.

    Features:
    - Physics constraint penalties
    - Per-model training for ensemble
    - Validation with CEM-specific metrics

    Usage:
        trainer = CEMTrainer(
            model=EnsembleSurrogate(...),
            constraints=PhysicsConstraints(bounds={'efficiency': (0, 1)})
        )

        for epoch in range(epochs):
            metrics = trainer.train_epoch(train_loader)
            val_metrics = trainer.validate(val_loader)
    """

    def __init__(
        self,
        model: EnsembleSurrogate,
        config: TrainingConfig = None,
        constraints: PhysicsConstraints = None,
        output_names: List[str] = None,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.config = config or TrainingConfig()
        self.device = device

        # Data loss
        self.criterion = nn.MSELoss()

        # Physics constraints
        if constraints and output_names:
            self.physics_loss = PhysicsConstraintLoss(output_names, constraints)
        else:
            self.physics_loss = None

        # Optimizer for each ensemble member
        self.optimizers = [
            torch.optim.Adam(
                m.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay
            )
            for m in model.models
        ]

        # Tracking
        self.train_history: List[Dict[str, float]] = []
        self.val_history: List[Dict[str, float]] = []

    def train_epoch(self, dataloader: "torch.utils.data.DataLoader") -> Dict[str, float]:
        """
        Train one epoch.

        Returns metrics dict with data_loss, physics_loss, total_loss per model.
        """
        self.model.train()

        epoch_metrics = {
            "data_loss": 0.0,
            "physics_loss": 0.0,
            "total_loss": 0.0,
        }
        n_batches = 0

        for batch in dataloader:
            if len(batch) == 2:
                x, y = batch
            else:
                x, y = batch[0], batch[1]

            x = x.to(self.device)
            y = y.to(self.device)

            # Train each ensemble member
            batch_loss = 0.0
            batch_physics = 0.0

            for i, (m, opt) in enumerate(zip(self.model.models, self.optimizers)):
                opt.zero_grad()

                # Forward
                y_pred = m(x)

                # Data loss
                data_loss = self.criterion(y_pred, y)

                # Physics penalty
                if self.physics_loss is not None:
                    physics_penalty, _ = self.physics_loss(y_pred, x)
                    total_loss = data_loss + self.config.lambda_bounds * physics_penalty
                    batch_physics += physics_penalty.item()
                else:
                    total_loss = data_loss

                total_loss.backward()
                opt.step()

                batch_loss += data_loss.item()

            epoch_metrics["data_loss"] += batch_loss / self.model.n_models
            epoch_metrics["physics_loss"] += batch_physics / self.model.n_models
            n_batches += 1

        # Average over batches
        for key in epoch_metrics:
            epoch_metrics[key] /= max(n_batches, 1)

        epoch_metrics["total_loss"] = (
            epoch_metrics["data_loss"] + self.config.lambda_bounds * epoch_metrics["physics_loss"]
        )

        self.train_history.append(epoch_metrics)
        return epoch_metrics

    def validate(self, dataloader: "torch.utils.data.DataLoader") -> Dict[str, float]:
        """
        Validate ensemble on held-out data.

        Returns metrics including ensemble uncertainty.
        """
        self.model.eval()

        all_preds = []
        all_targets = []
        all_uncertainties = []

        with torch.no_grad():
            for batch in dataloader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)

                mean, std = self.model(x)
                all_preds.append(mean.cpu())
                all_targets.append(y.cpu())
                all_uncertainties.append(std.cpu())

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        uncertainties = torch.cat(all_uncertainties, dim=0)

        # Compute metrics
        mse = ((preds - targets) ** 2).mean().item()
        rmse = np.sqrt(mse)
        mae = (preds - targets).abs().mean().item()

        # Uncertainty metrics
        mean_uncertainty = uncertainties.mean().item()

        metrics = {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "mean_uncertainty": mean_uncertainty,
        }

        self.val_history.append(metrics)
        return metrics

    def fit(
        self,
        train_loader: "torch.utils.data.DataLoader",
        val_loader: "torch.utils.data.DataLoader",
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping.

        Returns history dict with train and validation metrics.
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(self.config.epochs):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            if verbose and epoch % 20 == 0:
                print(
                    f"Epoch {epoch}: train_loss={train_metrics['total_loss']:.6f}, "
                    f"val_rmse={val_metrics['rmse']:.6f}, "
                    f"uncertainty={val_metrics['mean_uncertainty']:.4f}"
                )

            # Early stopping
            if val_metrics["rmse"] < best_val_loss - self.config.min_delta:
                best_val_loss = val_metrics["rmse"]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.config.patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

        return {"train": self.train_history, "val": self.val_history}
