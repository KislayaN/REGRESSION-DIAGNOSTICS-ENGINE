import numpy as np

class Lasso:
    def __init__(self, fit_intercept=True, alpha=0.001, epochs=None, tol=1e-6):
        self.fit_intercept = fit_intercept
        self.n_iter = epochs
        self.leaning_rate = alpha
        self.alpha=alpha
        self.tol = tol
        self.cost_history = []
        
    def _add_intercept(self, X):
        return np.column_stack((np.ones(len(X)), X))
        
    def fit(self, X, y):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        X_val = self._add_intercept(X_val) if self.fit_intercept else X_val
        
        n, p = X_val.shape
        self.weights = np.zeros(p)
        y = np.asarray(y_val).reshape(-1)
        
        for epoch in range(self.n_iter):
            prediction = X_val @ self.weights
            error = prediction - y
            
            cost = (1/n) * np.sum(error ** 2)
            self.cost_history.append(cost)
            
            if self.fit_intercept:
                l1_penalty = np.sign(self.weights)
                l1_penalty[0] = 0
            
            del_J = (2/n) * (X_val.T @ error) + (self.alpha * l1_penalty)
            
            if np.linalg.norm(del_J) < self.tol:
                break
            
            self.weights -= self.leaning_rate * del_J
            
        return self.weights
            
    def predict(self, X):
        X_val = X.values if hasattr(X, 'values') else X
            
        X_val = self._add_intercept(X_val) if self.fit_intercept else X_val
        prediction = X_val @ self.weights if self.fit_intercept else X_val @ self.weights
        
        return prediction
        