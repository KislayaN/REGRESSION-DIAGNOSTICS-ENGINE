import numpy as np

class OLS_Grad:
    def __init__(self, epochs=1000, learning_rate=0.001, tol=1e-6, fit_intercept=True):
        self.n_iter = epochs
        self.learning_rate = learning_rate
        self.weights = None
        self.cost_history = []
        self.tol = tol
        self.fit_intercept = fit_intercept
        self.mse = None
        self.is_diverged = False
        
    def _add_intercept(self, X):
        self.intercept_added = True
        return np.column_stack((np.ones(len(X)), X))
    
    def fit(self, X, y):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        X_val = self._add_intercept(X_val) if self.fit_intercept else X_val
            
        n, p = X_val.shape
        self.weights = np.zeros(p)
        y = np.asarray(y_val).reshape(-1)
        
        max_float = np.finfo(np.float64).max
        limit = np.sqrt(max_float / n)
        
        for epoch in range(self.n_iter):
            self.max_epochs = epoch
            prediction = X_val @ self.weights
            error = prediction - y
            
            if np.any(np.abs(error) > limit):
                self.is_diverged = True
                break
            
            cost = (1/n) * np.sum(error ** 2)
            
            self.cost_history.append(cost)
            
            # J = (1/n) * (((X_b @ self.weights) - y) ** 2) --------
            #                                                      |
            # J = (1/n) * (((X_b @ self.weights) - y) ** 2) <------ (after putting values)
            
            del_J = (2/n) * (X_val.T @ error)
             
            if np.linalg.norm(del_J) < self.tol:
                break
            
            self.weights -= self.learning_rate * del_J
        self.mse = self.cost_history[-1]
            
        if self.fit_intercept:
            self.weights = self.weights
            self.intercept = self.weights[0]
            self.coefficients = self.weights[1: ]
        else:
            self.intercept = 0.0
            self.coefficients = self.weights
            self.weights = self.coefficients
            
    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("run .fit() first")
        
        X_val = X.values if hasattr(X, 'values') else X
        X_val = self._add_intercept(X_val) if self.fit_intercept else X_val 
        
        if self.fit_intercept: 
            return X_val @ self.weights
        
        return X_val @ self.coefficients