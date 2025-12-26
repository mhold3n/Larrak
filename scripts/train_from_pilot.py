#!/usr/bin/env python3
"""Train structural surrogate from pilot DOE data.

Converts JSON results to training format and trains ensemble model.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import torch

from Simulations.hifi.training_schema import TrainingDataset, TrainingRecord
from truthmaker.surrogates.models.hifi_surrogates import StructuralSurrogate

# Load pilot results
data_file = Path("data/pilot_doe/pilot_results_20251225_220335.json")
with open(data_file) as f:
    results = json.load(f)

# Convert to training records (structural only)
records = []
for r in results:
    if r["solver"] == "structural" and r["success"]:
        p = r["params"]
        cal = r["outputs"]["calibration"]
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
        records.append(rec)
        print(f"Case {r['case_id']}: σ={cal['von_mises_max_mpa']:.1f} MPa")

print(f"\nTotal training records: {len(records)}")

# Create dataset
ds = TrainingDataset(records)
X, y = ds.get_structural_data()
print(f"X shape: {X.shape}, y shape: {y.shape}")

# Train with smaller ensemble for limited data
model = StructuralSurrogate(n_models=3)

# Train with more epochs for small dataset
print("\nTraining structural surrogate...")
for i, member in enumerate(model.models):
    optimizer = torch.optim.Adam(member.parameters(), lr=0.01)
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)

    for epoch in range(200):
        optimizer.zero_grad()
        pred = member(X_t)
        loss = torch.nn.functional.mse_loss(pred, y_t)
        loss.backward()
        optimizer.step()

    print(f"  Member {i + 1}/3: final_loss={loss.item():.6f}")

# Test predictions
mean, std = model.predict(X)
print(f"\nPredictions vs actual:")
for i in range(len(X)):
    print(f"  Case {i}: actual={y[i, 0]:.1f}, pred={mean[i, 0]:.1f} ± {std[i, 0]:.1f}")

# Save model
output_dir = Path("models/hifi")
output_dir.mkdir(parents=True, exist_ok=True)
model_path = output_dir / "structural_surrogate.pt"
model.save(str(model_path))
print(f"\nSaved model to {model_path}")
