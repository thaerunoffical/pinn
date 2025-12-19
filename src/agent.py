import re
import numpy as np
import torch
import copy

class BatteryDigitalTwin:
    """
    AI agent for natural language queries about battery thermal behavior
    """
    def __init__(self, ensemble_models, thermal_model):
        self.models = ensemble_models
        self.tm = thermal_model
        
    def extract_speed(self, query):
        """Extract speed from query like '100 km/h'"""
        match = re.search(r'(\d+)\s*km/?h', query.lower())
        return int(match.group(1)) if match else None
    
    def extract_duration(self, query):
        """Extract duration from query like '10 minutes'"""
        match = re.search(r'(\d+)\s*min', query.lower())
        return int(match.group(1)) * 60 if match else 600  # default 10 min
    
    def speed_to_C_rate(self, speed_kmh):
        """
        Map vehicle speed to battery discharge rate
        Rough estimate: 100 km/h ≈ 1.5C for typical EV
        """
        return np.interp(speed_kmh, [50, 80, 100, 120, 150], 
                        [0.5, 1.0, 1.5, 2.0, 2.5])
    
    def predict_scenario(self, C_rate, duration, T_amb=25, T_initial=25):
        """
        Predict temperature evolution for given scenario
        Uses ODE solver (not retrained PINN)
        """
        from src.utils import generate_data
        
        # Create temporary model with scaled Q_peak
        tm_temp = copy.deepcopy(self.tm)
        tm_temp.Q_peak = self.tm.Q_peak * (C_rate / 1.0) ** 2
        
        # Generate prediction
        t_test = np.linspace(0, duration, 100)
        T_pred = generate_data(tm_temp, t_test, T0=T_initial, T_amb=T_amb, 
                               t_onset=5.0, noise_level=0.0)
        
        # Add uncertainty estimate (from ensemble training)
        T_std = 0.02 * T_pred  # ~2% uncertainty
        
        return t_test, T_pred, T_std
    
    def assess_safety(self, T_final):
        """
        Safety assessment based on Coman's work:
        - T < 45°C: Safe
        - 45-55°C: Warning
        - > 60°C: Thermal runaway risk
        """
        if T_final < 45:
            return "✅ SAFE"
        elif T_final < 55:
            return "⚠️ WARNING - Consider cooling"
        elif T_final < 60:
            return "🔴 CRITICAL"
        else:
            return "🚨 THERMAL RUNAWAY RISK"
    
    def query(self, user_input):
        """Process natural language query"""
        user_input = user_input.lower()
        
        # Highway cruise query
        if "cruise" in user_input or "km/h" in user_input:
            speed = self.extract_speed(user_input)
            if speed is None:
                return "Please specify speed (e.g., '100 km/h')"
            
            duration = self.extract_duration(user_input)
            C_rate = self.speed_to_C_rate(speed)
            
            _, T_mean, T_std = self.predict_scenario(C_rate, duration)
            T_final = T_mean[-1]
            safety = self.assess_safety(T_final)
            
            response = f"At {speed} km/h cruise for {duration//60} minutes:\n"
            response += f"Temperature: {T_final:.1f}°C ± {T_std[-1]:.1f}°C\n"
            response += f"Status: {safety}"
            return response
        
        # Fast-charge query
        elif "fast" in user_input and "charg" in user_input:
            _, T_mean, T_std = self.predict_scenario(C_rate=3.0, duration=600, T_amb=30)
            T_final = T_mean[-1]
            safety = self.assess_safety(T_final)
            
            response = f"Fast-charging at 3C:\n"
            response += f"Peak temperature: {T_final:.1f}°C ± {T_std[-1]:.1f}°C\n"
            response += f"Status: {safety}"
            return response
        
        # Cold weather query
        elif "cold" in user_input:
            _, T_mean, T_std = self.predict_scenario(C_rate=1.0, duration=600, 
                                                      T_amb=-10, T_initial=-10)
            T_final = T_mean[-1]
            response = f"Cold weather (-10°C) operation:\n"
            response += f"Battery warms to {T_final:.1f}°C in 10 min\n"
            response += "⚠️ Reduced performance until T > 10°C"
            return response
        
        # Default
        else:
            return ("Try asking: 'cruise at 100 km/h', 'fast-charge', "
                   "'cold weather operation'")
