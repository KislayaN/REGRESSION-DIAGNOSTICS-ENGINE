import numpy as np
import pandas as pd

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.load_data import Load_data
from helpers.handle_missing_values import Imputer

class Validate(Load_data):
    def __init__(self):
        super().__init__()
        self.loader = Load_data()
        self.imputer = Imputer()
        self.is_validated = False
        self.contains_emptyvals = False
    
    def validate(self):
        if not self.data_loaded:
            raise ValueError("Run .load_data() first")
        
        else:
            # Check if X and y exists
            if self.X is None:
                raise ValueError("X is not loaded yet")
            if self.y is None:
                raise ValueError("y is not loaded yet")
            
            # Check if number of samples are same
            if self.X.shape[0] != self.y.shape[0]:
                raise ValueError("Sample count inconsistency present (same number of samples are valid)")
            
            # Check if y is 1D
            if self.y.ndim != 1:
                raise ValueError("target dimensions should be (1, ) or (1)")
            
            # Check if dataset has non numeric values
            if not np.issubdtype(self.X.dtype, np.number):
                raise ValueError("Non-Numeric values are present in dataset, only numeric values accepted")
            if not np.issubdtype(self.y.dtype, np.number):
                raise ValueError("Non-Numeric values are present in target, only numeric values accepted")
            
            # Check no missing values
            if np.isnan(self.X).any() or np.isnan(self.y).any():
                self.contains_emptyvals = True
                raise ValueError("Missing values detected")
            
            # Empty dataset
            if len(self.X) == 0:
                raise ValueError("Dataset is empty")