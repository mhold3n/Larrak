
import torch
import torch.nn as nn

class EngineSurrogateModel(nn.Module):
    """
    Multi-Layer Perceptron for predicting engine performance.
    
    Inputs: 
        - Normalized [rpm, p_int_bar, fuel_mass_mg]
    
    Outputs:
        - Normalized [thermal_efficiency, p_max_bar, abs_work_net_j]
    """
    def __init__(self, input_dim: int = 3, output_dim: int = 3, hidden_dim: int = 64):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.net(x)
