import torch
import numpy as np
from scipy.integrate import odeint

def generate_data(thermal_model, t_span, T0=25.0, T_amb=25.0, t_onset=10.0, noise_level=0.0):
    """
    Generate ground truth data using ODE solver
    
    Solves: m*Cp*dT/dt = Q_TR(t) - Q_conv
    """
    def ode_func(T, t):
        t_arr = np.atleast_1d(t)
        Q_TR = thermal_model.Q_TR(t_arr, t_onset)[0]
        Q_conv = thermal_model.h * thermal_model.A * (T - T_amb)
        dT_dt = (Q_TR - Q_conv) / (thermal_model.m * thermal_model.Cp)
        return dT_dt
    
    T_clean = odeint(ode_func, T0, t_span).flatten()
    
    if noise_level > 0:
        noise = np.random.normal(0, noise_level * T_clean.std(), len(T_clean))
        return T_clean + noise
    
    return T_clean


def train_pinn(model, thermal_model, t_data, T_data, t_physics, 
               epochs=5000, lr=1e-3, T_amb=25.0, t_onset=10.0, verbose=True):
    """
    Train PINN with hybrid optimizer: Adam → L-BFGS
    
    Loss = MSE_data + w_physics * MSE_physics + w_ic * MSE_ic
    """
    # Initial condition
    T0 = T_data[0]
    
    # Convert to tensors
    t_data_tensor = torch.tensor(t_data.reshape(-1, 1), dtype=torch.float32)
    T_data_tensor = torch.tensor(T_data.reshape(-1, 1), dtype=torch.float32)
    t_ic = torch.tensor([[0.0]], dtype=torch.float32)
    
    # Phase 1: Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Data loss
        T_pred = model(t_data_tensor)
        loss_data = torch.mean((T_pred - T_data_tensor)**2)
        
        # Physics loss
        residual = thermal_model.physics_residual(model, t_physics, T_amb, t_onset)
        loss_physics = torch.mean(residual**2)
        
        # Initial condition loss
        T_ic_pred = model(t_ic)
        loss_ic = torch.mean((T_ic_pred - T0)**2)
        
        # Total loss (balanced weights)
        loss = loss_data + 0.1 * loss_physics + loss_ic
        
        loss.backward()
        optimizer.step()
        
        if verbose and epoch % 1000 == 0:
            print(f"Epoch {epoch}: Data={loss_data.item():.6f}, Physics={loss_physics.item():.6f}")
    
    # Phase 2: L-BFGS fine-tuning
    optimizer = torch.optim.LBFGS(model.parameters(), lr=0.1, max_iter=20)
    
    def closure():
        optimizer.zero_grad()
        T_pred = model(t_data_tensor)
        loss_data = torch.mean((T_pred - T_data_tensor)**2)
        residual = thermal_model.physics_residual(model, t_physics, T_amb, t_onset)
        loss_physics = torch.mean(residual**2)
        T_ic_pred = model(t_ic)
        loss_ic = torch.mean((T_ic_pred - T0)**2)
        loss = loss_data + 0.1 * loss_physics + loss_ic
        loss.backward()
        return loss
    
    for _ in range(30):
        optimizer.step(closure)
    
    return model


def train_ensemble(n_models, thermal_model, t_data, T_data, t_physics, 
                   noise_levels=[0.0, 0.01, 0.02], **kwargs):
    """Train ensemble of PINNs with different noise realizations"""
    from src.model import BatteryThermalPINN
    
    ensemble = []
    for i in range(n_models):
        print(f"Training model {i+1}/{n_models}...", end=" ")
        
        model = BatteryThermalPINN()
        torch.manual_seed(42 + i)
        
        # Add random noise to data
        noise_level = np.random.choice(noise_levels)
        T_noisy = T_data + np.random.normal(0, noise_level * T_data.std(), len(T_data))
        
        model = train_pinn(model, thermal_model, t_data, T_noisy, t_physics, 
                          verbose=False, **kwargs)
        ensemble.append(model)
        print("✓")
    
    print("Ensemble training complete!")
    return ensemble


def predict_with_uncertainty(ensemble_models, t_test):
    """Predict with uncertainty quantification using ensemble"""
    predictions = []
    for model in ensemble_models:
        with torch.no_grad():
            t_tensor = torch.tensor(t_test.reshape(-1, 1), dtype=torch.float32)
            predictions.append(model(t_tensor).numpy().flatten())
    
    predictions = np.array(predictions)
    return predictions.mean(axis=0), predictions.std(axis=0)
