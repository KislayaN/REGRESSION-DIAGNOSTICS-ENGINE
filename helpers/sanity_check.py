import numpy as np
import pandas as pd

class Sanity_Check:
    def __init__(self):
        self.warnings = []
        
    def sanity_check(self, X, y):
        self.X = X
        self.target = y
        self.y = self.X[self.target].values
        
        # Check if target variance is low
        if np.var(self.y) < 1e-6:
            self.warnings.append("Target has very low variance")
        
        # Check correlation between target and features
        corr = X.corr()[self.target].drop(self.target)
        if (corr.abs() < 0.05).all():
            self.warnings.append("Target weakly correlated with other features")
            
        # Check if target is accidentally inside features
        for col in self.X.columns:
            if col != self.target and np.array_equal(self.X[col].values, self.target):
                self.warnings.append(f"Target appears in the features {col}")
                
        # Check if features are constant
        feature_var = self.X.drop(columns=[self.target]).var()
        if (feature_var < 1e-6).any():
            self.warnings.append("Some features have near zero variance")
            
        return self.warningsgit 