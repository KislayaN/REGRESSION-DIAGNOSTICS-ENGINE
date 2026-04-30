import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from evaluation_metrics.metrics import Get_metrics

class Performance:
    def __init__(self):
        self.get_metrics = Get_metrics()
        self.insights = []
        
    def compare_mse(self, mse_dict):
        """
        mse_dict = {
            "ols":mse_ols,
            "gd":mse_gd,
            "lasso":mse_lasso,
            "ridge":mse_ridge,
            "lr":mse_lr
        }
        """
        min_val = min(mse_dict.values())
        key_min_val = [k for k, v in mse_dict.items() if v == min_val]
        
        self.insights.append(f"The minimum mse value is {min_val:.5f} for {key_min_val}")
        
        return self.insights
    
    def compare_r2(self, r2_dict):
        """
        r2_dict = {
            "ols":r2_ols,
            "gd":r2_gd,
            "lasso":r2_lasso,
            "ridge":r2_ridge,
            "lr":r2_lr
        }
        """
        max_val = max(r2_dict.values())
        key_max_val = [k for k, v in r2_dict.items() if v == max_val]
        
        self.insights.append(f"{key_max_val} captures the most variance {max_val:.5f}")
        
        return self.insights