import numpy as np

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
        self.results = {}
        
    def compare_mse(self, y_true, predictions_dict):
        """
        predictions_dict = {
            "ols":y_pred_ols,
            "gd":y_pred_gd,
            "lasso":y_pred_lasso,
            "ridge":y_pred_ridge,
            "lr":y_pred_lr
        }
        """
        for name, y_pred in predictions_dict.items():
            self.results[name] = self.get_metrics.mse(y_true, y_pred)
        return self.results