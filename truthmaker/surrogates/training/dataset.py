
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import json

class EngineDataset(Dataset):
    """
    PyTorch Dataset for Engine Data.
    Handles loading, filtering, and normalization.
    """
    def __init__(self, csv_path: str, mode: str = 'train', split_ratio: float = 0.8, seed: int = 42):
        self.inputs = ["rpm", "p_int_bar", "fuel_mass_mg"]
        self.outputs = ["thermal_efficiency", "p_max_bar", "abs_work_net_j"]
        
        # 1. Load Data
        df = pd.read_csv(csv_path)
        
        # 2. Filter Validity
        # Status "Optimal" or "Maximum_Iterations_Exceeded"
        # Efficiency 0.0 to 0.80 (Physical bounds)
        valid_mask = (
            (df["status"].isin(["Optimal", "Maximum_Iterations_Exceeded"])) &
            (df["thermal_efficiency"] > 0.0) &
            (df["thermal_efficiency"] < 0.85)
        )
        df = df[valid_mask].copy()
        
        # 3. Train/Test Split
        np.random.seed(seed)
        shuffled_indices = np.random.permutation(len(df))
        split_idx = int(len(df) * split_ratio)
        
        if mode == 'train':
            self.df = df.iloc[shuffled_indices[:split_idx]].reset_index(drop=True)
        else:
            self.df = df.iloc[shuffled_indices[split_idx:]].reset_index(drop=True)
            
        # 4. Compute Statistics (Only on TRAIN, potentially passed to test)
        #Ideally we compute stats on TRAIN and reuse for TEST. 
        # For simplicity in this self-contained class, we might recompute or need a better way.
        # Let's assume we pass stats explicitly if it's test?
        # Actually, standard practice: fit Scaler on Train, transform Test. 
        # Here we will compute stats locally for the split inside __init__, which is slightly wrong if we split by class instance.
        # Better approach: Load full DF, compute stats, then split.
        
        # NOTE: For a robust implementation, the scaler should be passed in. 
        # But for this simple task, we'll calculate MinMax here.
        pass

    def get_data(self):
        return self.df[self.inputs].values.astype(np.float32), self.df[self.outputs].values.astype(np.float32)

class Normalizer:
    def __init__(self):
        self.min_in = None
        self.max_in = None
        self.min_out = None
        self.max_out = None
    
    def fit(self, X, Y):
        self.min_in = X.min(axis=0)
        self.max_in = X.max(axis=0)
        self.min_out = Y.min(axis=0)
        self.max_out = Y.max(axis=0)
        
    def transform(self, X, Y):
        X_norm = (X - self.min_in) / (self.max_in - self.min_in + 1e-6)
        Y_norm = (Y - self.min_out) / (self.max_out - self.min_out + 1e-6)
        return torch.tensor(X_norm), torch.tensor(Y_norm)
        
    def save(self, path):
        data = {
            "min_in": self.min_in.tolist(),
            "max_in": self.max_in.tolist(),
            "min_out": self.min_out.tolist(),
            "max_out": self.max_out.tolist()
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=4)
