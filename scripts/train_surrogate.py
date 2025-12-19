
import os
import sys

# Add project root to path
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
import torch.optim as optim
from surrogate.model import EngineSurrogateModel
from surrogate.dataset import EngineDataset, Normalizer
import numpy as np

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

def train():
    # Config
    # Default DOE results now live under dashboard/thermo
    CSV_PATH = os.environ.get(
        "LARRAK_THERMO_RESULTS",
        os.environ.get("LARRAK_PHASE1_RESULTS", r"dashboard/thermo/thermo_doe_results.csv"),
    )
    MODEL_PATH = r"surrogate/model_artifacts/surrogate_model.pth"
    SCALER_PATH = r"surrogate/model_artifacts/scaler_params.json"
    REPORT_PATH = r"surrogate/model_artifacts/training_report.png"
    make_report = str(os.environ.get("LARRAK_SURROGATE_PLOT", os.environ.get("LARRAK_INTERPRETER_PLOT", "0"))).strip().lower() in {"1", "true", "yes", "y", "on"}
    
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    # 1. Prepare Data
    print("Loading datasets...")
    train_ds_raw = EngineDataset(CSV_PATH, mode='train')
    test_ds_raw = EngineDataset(CSV_PATH, mode='test')
    
    X_train, Y_train = train_ds_raw.get_data()
    X_test, Y_test = test_ds_raw.get_data()
    
    # Normalize
    scaler = Normalizer()
    scaler.fit(X_train, Y_train)
    scaler.save(SCALER_PATH)
    
    tn_X, tn_Y = scaler.transform(X_train, Y_train)
    tt_X, tt_Y = scaler.transform(X_test, Y_test)
    
    # 2. Model Setup
    model = EngineSurrogateModel()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Training Loop
    epochs = int(os.environ.get("LARRAK_SURROGATE_EPOCHS", os.environ.get("LARRAK_INTERPRETER_EPOCHS", "200")))
    batch_size = 64
    
    train_losses = []
    test_losses = []
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        model.train()
        
        # Shuffle
        permutation = torch.randperm(tn_X.size()[0])
        
        epoch_loss = 0
        for i in range(0, tn_X.size()[0], batch_size):
            indices = permutation[i:i+batch_size]
            batch_x, batch_y = tn_X[indices], tn_Y[indices]
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        train_losses.append(epoch_loss / (len(tn_X) / batch_size))
        
        # Validation
        model.eval()
        with torch.no_grad():
            test_out = model(tt_X)
            t_loss = criterion(test_out, tt_Y)
            test_losses.append(t_loss.item())
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Train Loss {train_losses[-1]:.6f}, Test Loss {test_losses[-1]:.6f}")
            
    # 4. Evaluation
    print("Training complete. Evaluating...")
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    with torch.no_grad():
        preds = model(tt_X).numpy()

    if make_report:
        # Plotting is optional because matplotlib/GUI backends can introduce Windows teardown issues.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 5))
        
        # Loss Curve
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train')
        plt.plot(test_losses, label='Test')
        plt.title('Loss History')
        plt.legend()
        plt.yscale('log')
        
        # Parity Plot (Thermal Efficiency)
        plt.subplot(1, 2, 2)
        plt.scatter(tt_Y[:, 0], preds[:, 0], alpha=0.1, s=1)
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel('Actual (Norm)')
        plt.ylabel('Predicted (Norm)')
        plt.title('Parity: Thermal Efficiency')
        
        plt.tight_layout()
        plt.savefig(REPORT_PATH)
        print(f"Report saved to {REPORT_PATH}")
    
    # Calculate R2
    r2_eff = _r2_score_np(tt_Y[:, 0], preds[:, 0])
    print(f"Test R2 Score (Efficiency): {r2_eff:.4f}")

if __name__ == "__main__":
    train()
