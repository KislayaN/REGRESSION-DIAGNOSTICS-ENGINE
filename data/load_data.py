import numpy as np

import sys
import os

current_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.exists(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)
    
from helpers.handle_missing_values import Imputer

class Load_data:
    is_1D = False
    contains_same_samples = False
    is_numeric = False
    validated = False
    
    def __init__(self, data=None, target=None):
        self.dataset = data
        self.target = target
        self.imputer = Imputer()
        
    def validate_target(self):
        # validation 1: target vector must be a 1-Dimensional array
        if self.target.ndim == 1:
            self.is_1D = True    
            
        # validation 2: X and y should have same number of samples
        if self.dataset.shape[1] == self.target.shape[1]:
            self.contains_same_samples = True
            
        # validation 3: y should be numeric
        if np.issubdtype(self.target, np.number):
            self.is_numeric = True
        
        # validation 4: should not have missing values, if present (handle them)
        if self.target.isna().any().any():
            target = self.imputer.transform(self.target)
            
        self.validated = True
        return self.validated
            
    def get_data(self):
        """Returns a tuple of dataset and target"""
        
        if self.validated:
            return self.dataset, self.target
        
        return "The target need to validate, call .validate_target() first"