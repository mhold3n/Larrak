
import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
import json
import pandas as pd
from truthmaker.surrogates.models.model import EngineSurrogateModel
from truthmaker.surrogates.training.dataset import EngineDataset, Normalizer

def _r2_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Minimal R^2 implementation to avoid importing sklearn (can trigger OpenMP DLL conflicts on Windows).
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)

def validate():
    MODEL_PATH = r"surrogate/model_artifacts/surrogate_model.pth"
    SCALER_PATH = r"surrogate/model_artifacts/scaler_params.json"
    CSV_PATH = os.environ.get(
        "LARRAK_THERMO_RESULTS",
        os.environ.get("LARRAK_PHASE1_RESULTS", r"output/thermo/thermo_doe_results.csv"),
    )
    r2_target = float(os.environ.get("LARRAK_SURROGATE_R2_TARGET", os.environ.get("LARRAK_INTERPRETER_R2_TARGET", "0.95")))

    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    # 1. Load Data
    test_ds = EngineDataset(CSV_PATH, mode='test')
    X_test, Y_test = test_ds.get_data()

    # 2. Load Scaler
    with open(SCALER_PATH, 'r') as f:
        scaler_data = json.load(f)
    
    # Reconstruct Scaler
    class LoadedNormalizer(Normalizer):
        def __init__(self, data):
            self.min_in = np.array(data["min_in"])
            self.max_in = np.array(data["max_in"])
            self.min_out = np.array(data["min_out"])
            self.max_out = np.array(data["max_out"])
            
    scaler = LoadedNormalizer(scaler_data)
    
    # 3. Predict
    # Norm Input
    X_norm, _ = scaler.transform(X_test, Y_test) # Y ignored transform
    
    # Load Model
    model = EngineSurrogateModel()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    
    with torch.no_grad():
        preds_norm = model(X_norm.float()).numpy() # Explicit cast to float32
        
    # Unscale Output? 
    # Let's check R2 on normalized space first (it's the same R2).
    # Inputs: [rpm, p_int, fuel]
    # Outputs: [eff, p_max, work]
    
    _, Y_norm = scaler.transform(X_test, Y_test)
    Y_norm_np = Y_norm.numpy()
    
    r2_eff = _r2_score_np(Y_norm_np[:, 0], preds_norm[:, 0])
    r2_pmax = _r2_score_np(Y_norm_np[:, 1], preds_norm[:, 1])
    r2_work = _r2_score_np(Y_norm_np[:, 2], preds_norm[:, 2])
    
    print("-" * 30)
    print(f"Surrogate Validation Results")
    print("-" * 30)
    print(f"Test Set Size: {len(X_test)}")
    print(f"R2 Efficiency: {r2_eff:.4f}")
    print(f"R2 Peak Press: {r2_pmax:.4f}")
    print(f"R2 Work Out  : {r2_work:.4f}")
    print("-" * 30)
    
    if (r2_eff >= r2_target) and (r2_pmax >= r2_target) and (r2_work >= r2_target):
        print(f"SUCCESS: Model meets accuracy target (>= {r2_target:.2f}).")
        sys.exit(0)
    else:
        print(f"WARNING: Model needs tuning (target >= {r2_target:.2f}).")
        sys.exit(2)

if __name__ == "__main__":
    validate()
