#!/usr/bin/env python3
"""Train both structural and thermal surrogates from pilot DOE data."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from Simulations.hifi.training_schema import TrainingDataset, TrainingRecord
from truthmaker.surrogates.models.hifi_surrogates import StructuralSurrogate, ThermalSurrogate


def train_surrogate(model, X, y, epochs=200, lr=0.01):
    """Train all ensemble members."""
    for i, member in enumerate(model.models):
        optimizer = torch.optim.Adam(member.parameters(), lr=lr)
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        for epoch in range(epochs):
            optimizer.zero_grad()
            pred = member(X_t)
            loss = torch.nn.functional.mse_loss(pred, y_t)
            loss.backward()
            optimizer.step()

        print(f"  Member {i + 1}/{len(model.models)}: final_loss={loss.item():.6f}")


def main():
    # Find latest results file
    data_dir = Path("data/pilot_doe")
    results_files = sorted(data_dir.glob("pilot_results_*.json"))
    if not results_files:
        print("No pilot results found!")
        return 1

    data_file = results_files[-1]  # Latest
    print(f"Loading: {data_file}")

    with open(data_file) as f:
        results = json.load(f)

    # Build training records
    struct_records, thermal_records = [], []

    for r in results:
        if not r["success"]:
            continue

        p = r["params"]
        cal = r["outputs"]["calibration"]

        if r["solver"] == "structural":
            rec = TrainingRecord(
                case_id=r["case_id"],
                bore=p["bore_mm"],
                stroke=p["stroke_mm"],
                cr=p["compression_ratio"],
                rpm=p["rpm"],
                load=p["load_fraction"],
                von_mises_max=cal["von_mises_max_mpa"],
                solver_success=True,
            )
            struct_records.append(rec)

        elif r["solver"] == "thermal":
            rec = TrainingRecord(
                case_id=r["case_id"],
                bore=p["bore_mm"],
                stroke=p["stroke_mm"],
                cr=p["compression_ratio"],
                rpm=p["rpm"],
                load=p["load_fraction"],
                T_crown_max=cal["T_crown_max_K"],
                solver_success=True,
            )
            thermal_records.append(rec)

    print(f"Structural records: {len(struct_records)}")
    print(f"Thermal records: {len(thermal_records)}")

    output_dir = Path("models/hifi")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train structural surrogate
    if struct_records:
        print("\n=== Training Structural Surrogate ===")
        ds = TrainingDataset(struct_records)
        X, y = ds.get_structural_data()
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        model = StructuralSurrogate(n_models=3)
        train_surrogate(model, X, y)

        model_path = output_dir / "structural_surrogate.pt"
        model.save(str(model_path))
        print(f"Saved: {model_path}")

    # Train thermal surrogate
    if thermal_records:
        print("\n=== Training Thermal Surrogate ===")
        ds = TrainingDataset(thermal_records)
        X, y = ds.get_thermal_data()
        print(f"X shape: {X.shape}, y shape: {y.shape}")

        model = ThermalSurrogate(n_models=3)
        train_surrogate(model, X, y)

        model_path = output_dir / "thermal_surrogate.pt"
        model.save(str(model_path))
        print(f"Saved: {model_path}")

    print("\n=== Training Complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
