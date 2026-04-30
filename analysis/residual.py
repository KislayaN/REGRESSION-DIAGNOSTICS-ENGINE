import numpy as np

class Resid_insights:
    def __init__(self):
        self.insights = []
    
    def analyze_residuals(self, residual_mean_dict):
        
        for key, value in residual_mean_dict.items():
            if abs(value) < 1e-3:
                self.insights.append(f"Residuals are centered round zero for {key}")
            else: 
                self.insights.append(f"Residuals are biased (mean not zero) for {key}")
        
            if np.std(value) > 10:
                self.insights.append(f"High variance present in {key}")
        
        return self.insights