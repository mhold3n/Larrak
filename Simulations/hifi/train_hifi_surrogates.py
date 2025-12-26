"""Trainer for HiFi Surrogate Models.

Trains ensemble models on DOE data from CFD/FEA simulations.

Usage:
    python train_hifi_surrogates.py --data data/doe_results.json --output models/
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add parent to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from Simulations.hifi.training_schema import NormalizationParams, TrainingDataset
from truthmaker.surrogates.models.hifi_surrogates import (
    FlowCoefficientSurrogate,
    StructuralSurrogate,
    ThermalSurrogate,
)


def train_ensemble(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 32,
    lr: float = 1e-3,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Train ensemble model with early stopping.

    Trains each ensemble member on bootstrapped samples for diversity.
    """
    n_models = len(model.models)
    history = {"train_loss": [], "val_loss": []}

    criterion = nn.MSELoss()

    for i, member in enumerate(model.models):
        # Bootstrap sample for this member
        n_samples = len(X_train)
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X_train[indices]
        y_boot = y_train[indices]

        train_loader = DataLoader(
            TensorDataset(
                torch.tensor(X_boot, dtype=torch.float32), torch.tensor(y_boot, dtype=torch.float32)
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        optimizer = optim.Adam(member.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(epochs):
            member.train()
            train_losses = []

            for X_batch, y_batch in train_loader:
                optimizer.zero_grad()
                pred = member(X_batch)
                loss = criterion(pred, y_batch)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            # Validation
            member.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32)
                y_val_t = torch.tensor(y_val, dtype=torch.float32)
                val_pred = member(X_val_t)
                val_loss = criterion(val_pred, y_val_t).item()

            scheduler.step(val_loss)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break

        if verbose:
            print(f"  Member {i + 1}/{n_models}: val_loss={best_val_loss:.6f}")

    # Final validation on full ensemble
    model.eval()
    with torch.no_grad():
        X_val_t = torch.tensor(X_val, dtype=torch.float32)
        y_val_t = torch.tensor(y_val, dtype=torch.float32)
        mean, std = model(X_val_t)
        ensemble_loss = criterion(mean, y_val_t).item()

    if verbose:
        print(f"  Ensemble val_loss={ensemble_loss:.6f}, mean_std={std.mean():.4f}")

    history["final_val_loss"] = ensemble_loss
    history["mean_uncertainty"] = std.mean().item()

    return history


def train_all_surrogates(
    data_path: str, output_dir: str, epochs: int = 100, n_models: int = 5
) -> dict[str, dict]:
    """Train all surrogate types on DOE data."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    if data_path.endswith(".parquet"):
        dataset = TrainingDataset.from_parquet(data_path)
    else:
        dataset = TrainingDataset.from_json(data_path)

    print(f"Loaded {len(dataset.records)} training records")

    # Split
    train_ds, val_ds = dataset.split(train_frac=0.8)

    results = {}

    # 1. Thermal surrogate
    print("\n=== Training Thermal Surrogate ===")
    X_train, y_train = train_ds.get_thermal_data()
    X_val, y_val = val_ds.get_thermal_data()

    if len(X_train) > 0:
        thermal_model = ThermalSurrogate(n_models=n_models)
        history = train_ensemble(thermal_model, X_train, y_train, X_val, y_val, epochs=epochs)

        model_path = output_dir / "thermal_surrogate.pt"
        thermal_model.save(str(model_path))
        print(f"Saved thermal model to {model_path}")
        results["thermal"] = history
    else:
        print("  No thermal data available")

    # 2. Structural surrogate
    print("\n=== Training Structural Surrogate ===")
    X_train, y_train = train_ds.get_structural_data()
    X_val, y_val = val_ds.get_structural_data()

    if len(X_train) > 0:
        structural_model = StructuralSurrogate(n_models=n_models)
        history = train_ensemble(structural_model, X_train, y_train, X_val, y_val, epochs=epochs)

        model_path = output_dir / "structural_surrogate.pt"
        structural_model.save(str(model_path))
        print(f"Saved structural model to {model_path}")
        results["structural"] = history
    else:
        print("  No structural data available")

    # 3. Flow coefficient surrogate
    print("\n=== Training Flow Surrogate ===")
    X_train, y_train = train_ds.get_flow_data()
    X_val, y_val = val_ds.get_flow_data()

    if len(X_train) > 0:
        flow_model = FlowCoefficientSurrogate(n_models=n_models)
        history = train_ensemble(flow_model, X_train, y_train, X_val, y_val, epochs=epochs)

        model_path = output_dir / "flow_surrogate.pt"
        flow_model.save(str(model_path))
        print(f"Saved flow model to {model_path}")
        results["flow"] = history
    else:
        print("  No flow data available")

    # Save normalization params
    dataset.norm_params.save(str(output_dir / "normalization.json"))

    # Save summary
    with open(output_dir / "training_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description="Train HiFi surrogates")
    parser.add_argument("--data", required=True, help="Path to DOE results (JSON or Parquet)")
    parser.add_argument("--output", default="models/hifi", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--n-models", type=int, default=5, help="Ensemble size")

    args = parser.parse_args()

    results = train_all_surrogates(
        data_path=args.data, output_dir=args.output, epochs=args.epochs, n_models=args.n_models
    )

    print("\n=== Training Complete ===")
    for name, hist in results.items():
        print(f"{name}: val_loss={hist.get('final_val_loss', 'N/A'):.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
