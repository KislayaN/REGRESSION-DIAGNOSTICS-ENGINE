import numpy as np

class Resid_insights:
    def __init__(self):
        self.residuals = None
        self.insights = []
    
    def analyze_residuals(self, y_true, y_pred):
        self.residuals = y_true - y_pred
        
        resid_mean = np.mean(self.residuals)
        
        if abs(resid_mean) < 1e-3:
            self.insights.append("Residual are centered around zero")
        else: 
            self.insights.append("Residuals are biased (mean not zero)")
        
        if np.std(self.residuals) > 10:
            self.insights.append("High variance present in residuals")
        
        return self.residuals ,self.insights