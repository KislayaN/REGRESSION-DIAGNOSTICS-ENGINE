import numpy as np

class Ridge:
    def __init__(self, fit_intercept=True, alpha=0.001):
        self.fit_intercept = fit_intercept
        self.alpha=alpha
        
    def _add_intercept(self, X):
        return np.column_stack((np.ones(len(X)), X))
        
    def fit(self, X, y):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        X_val = self._add_intercept(X_val) if self.fit_intercept else X_val
        
        n, p = X_val.shape
        Identity = np.eye(p)
        Identity[0, 0] = 0
        
        betas = np.linalg.pinv((X_val.T @ X_val) + (self.alpha * Identity)) @ X_val.T @ y_val
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