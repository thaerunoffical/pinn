import torch
import torch.nn as nn

class BatteryThermalPINN(nn.Module):
    """
    Physics-Informed Neural Network for battery thermal prediction
    
    Architecture:
        - Input: time (t)
        - 3 hidden layers × 32 neurons
        - Activation: tanh (common for PINNs)
        - Output: temperature (T)
    """
    def __init__(self, t_max=50.0):
        super().__init__()
        self.t_max = t_max  # for input normalization
        
        self.net = nn.Sequential(
            nn.Linear(1, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # Xavier initialization (helps convergence)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, t):
        # Normalize input to [0, 1] range
        t_norm = t / self.t_max
        return self.net(t_norm)
