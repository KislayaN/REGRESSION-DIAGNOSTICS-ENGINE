# This is the OLS estimator (BLUE)

import numpy as np

class OLS:
    def __init__(self, fit_intercept=False):
        self.fit_intercept = fit_intercept
        self.intercept_added = False
    
    def _add_intercept(self, X):
        self.intercept_added = True
        return np.column_stack((np.ones(len(X)), X))
    
    def fit(self, X, target):
        
        X_val = X.values if hasattr(X, 'values') else X
        target_val = target.values if hasattr(target, 'values') else target
        
        if self.intercept_added:
            return X_val
        else:
            X_val = self._add_intercept(X_val) if self.fit_intercept else X_val
            self.intercept_added = True
        
        betas = np.linalg.pinv(X_val.T @ X_val) @ X_val.T @ target_val
        self.betas = np.asarray(betas).flatten()
        
        if self.fit_intercept:
            self.coefficients = betas[1: ]
            self.intercept = betas[0]
        else: 
            self.coefficients = betas
            self.intercept = 0.0
            
    def predict(self, X):
        X_val = X.values if hasattr(X, 'values') else X
        
        X_val = self._add_intercept(X_val) if self.fit_intercept else X_val
        
        if self.coefficients is not None:
            prediction = X_val @ self.betas if self.fit_intercept else X_val @ self.coefficients
            return prediction
        raise ValueError("run .fit() first")