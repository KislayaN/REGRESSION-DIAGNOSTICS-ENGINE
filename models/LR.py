from sklearn.linear_model import LinearRegression

import numpy as np

class _LinearRegression:
    def __init__(self, fit_intercept=True):
        self.model = LinearRegression(fit_intercept=fit_intercept)
        
    def fit(self, X, y):
        X_val = X.values if hasattr(X, 'values') else X
        y_val = y.values if hasattr(y, 'values') else y
        
        self.model.fit(X, y)
        
        self.coefficients = self.model.coef_
        self.intercept = self.model.intercept_
        
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions