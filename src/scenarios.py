import numpy as np

class ScenarioGenerator:
    """
    Generate realistic battery operating scenarios
    
    Each scenario defines:
        - Ambient temperature
        - Heat generation scaling (based on C-rate)
        - Thermal runaway onset time
    """
    def __init__(self, thermal_model):
        self.tm = thermal_model
        
    def fast_charging_stress(self):
        """
        Fast-charging at 3C
        Heat generation scales as I² → 9x at 3C vs 1C
        """
        C_rate = 3.0
        Q_scale = (C_rate / 1.0) ** 2  # ~9x heat
        
        return {
            'name': 'Fast-Charging (3C)',
            'C_rate': C_rate,
            'T_amb': 35,  # Hot environment
            't_onset': 5.0,
            'Q_scale': Q_scale
        }
    
    def cold_weather_start(self):
        """Cold weather operation (-10°C ambient)"""
        return {
            'name': 'Cold Weather Start',
            'C_rate': 1.0,
            'T_amb': -10,
            'T_initial': -10,
            't_onset': 15.0,
            'Q_scale': 1.2
        }
    
    def manufacturing_defect(self):
        """Cell with slight internal short (defect)"""
        return {
            'name': 'Manufacturing Defect',
            'C_rate': 2.0,
            'T_amb': 30,
            't_onset': 8.0,
            'Q_scale': 1.3
        }
    
    def aging_battery(self):
        """Aged cell (reduced heat capacity)"""
        return {
            'name': 'Aging Battery (80% SOH)',
            'C_rate': 2.0,
            'T_amb': 25,
            't_onset': 10.0,
            'Q_scale': 1.1,
            'Cp_modified': 720  # Reduced Cp
        }
    
    def highway_cruise(self, speed_kmh=100, duration_min=10):
        """
        Highway cruise scenario
        Requested by Dr. Coman: "What happens at 100 km/h for 10 min?"
        
        Assumption: speed → discharge rate (rough linear mapping)
        """
        # Map speed to C-rate (empirical)
        C_rate = np.interp(speed_kmh, [50, 80, 100, 120], [0.5, 1.0, 1.5, 2.0])
        Q_scale = (C_rate / 1.0) ** 2
        
        return {
            'name': f'Highway Cruise ({speed_kmh} km/h)',
            'C_rate': C_rate,
            'T_amb': 25,
            't_duration': duration_min * 60,
            'Q_scale': Q_scale
        }
    
    def get_all_scenarios(self):
        return [
            self.fast_charging_stress(),
            self.cold_weather_start(),
            self.manufacturing_defect(),
            self.aging_battery(),
            self.highway_cruise(100, 10)
        ]
