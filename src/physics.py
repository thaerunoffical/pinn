import torch
import numpy as np
from torch.autograd import grad

class ThermalModel:
    """
    0D lumped thermal model for 18650 Li-ion cell
    Based on Coman et al. 2022, Eq. 1 (simplified)
    
    Governing equation:
        m * Cp * dT/dt = Q_TR(t) - Q_conv
    
    Where:
        Q_TR(t) = heat generation from thermal runaway (from Fig. 3)
        Q_conv = h * A * (T - T_amb) = convective heat loss
    """
    def __init__(self):
        # Cell properties (given by Dr. Coman)
        self.m = 0.042      # kg (42g for 18650)
        self.Cp = 800       # J/kg/K
        
        # Geometry (18650 cylindrical cell)
        self.d = 0.018      # m (diameter = 18mm)
        self.L = 0.065      # m (length = 65mm)
        self.A = np.pi * self.d * self.L  # m² (surface area ≈ 0.00368)
        
        # Convection (from Coman Table I - natural convection)
        self.h = 20         # W/m²/K
        
        # Thermal runaway parameters (from Coman Fig. 3)
        # Peak heat rate ~10,000 W, total energy ~32.2 kJ
        self.Q_peak = 10000  # W
        
    def Q_TR(self, t, t_onset=10.0):
        """
        Heat generation rate from thermal runaway
        Approximated from Coman et al. Fig. 3
        
        Shape: fast rise (0-1s) → peak → slower decay (1-6s)
        Total integrated energy ≈ 32.2 kJ
        """
        t_rel = t - t_onset
        mask = (t_rel >= 0) & (t_rel <= 6)
        Q = np.zeros_like(t)
        
        # Rising phase (0 to 1 second)
        rise_mask = (t_rel >= 0) & (t_rel <= 1.0)
        Q[rise_mask] = self.Q_peak * (1 - np.exp(-5 * t_rel[rise_mask]))
        
        # Decay phase (1 to 6 seconds)
        decay_mask = (t_rel > 1.0) & (t_rel <= 6.0)
        Q[decay_mask] = self.Q_peak * np.exp(-0.6 * (t_rel[decay_mask] - 1.0))
        
        return Q
    
    def Q_TR_torch(self, t_tensor, t_onset=10.0):
        """Torch version for automatic differentiation in PINN"""
        t_rel = t_tensor - t_onset
        Q = torch.zeros_like(t_tensor)
        
        rise_mask = (t_rel >= 0) & (t_rel <= 1.0)
        Q[rise_mask] = self.Q_peak * (1 - torch.exp(-5 * t_rel[rise_mask]))
        
        decay_mask = (t_rel > 1.0) & (t_rel <= 6.0)
        Q[decay_mask] = self.Q_peak * torch.exp(-0.6 * (t_rel[decay_mask] - 1.0))
        
        return Q
    
    def physics_residual(self, model, t, T_amb=25.0, t_onset=10.0):
        """
        Computes PDE residual for PINN training:
            Residual = m*Cp*dT/dt - Q_TR(t) + Q_conv
        
        Should be ≈ 0 when model satisfies physics
        """
        t_tensor = torch.tensor(t.reshape(-1, 1), dtype=torch.float32, requires_grad=True)
        T = model(t_tensor)
        
        # Compute dT/dt using automatic differentiation
        dT_dt = grad(T.sum(), t_tensor, create_graph=True)[0]
        
        # Heat generation
        Q_TR = self.Q_TR_torch(t_tensor, t_onset)
        
        # Convective loss
        Q_conv = self.h * self.A * (T - T_amb)
        
        # Physics residual (should be zero)
        residual = self.m * self.Cp * dT_dt - Q_TR + Q_conv
        
        return residual
