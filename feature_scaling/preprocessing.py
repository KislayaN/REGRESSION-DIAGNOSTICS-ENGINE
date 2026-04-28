import numpy as np

from sklearn.preprocessing import StandardScaler

class Preprocess:
    def __init__(self, perform_scaling=True):
        self.perform_scaling = perform_scaling
        self.scaler = StandardScaler()
        
    def fit(self, X):
        self.scalable_cols = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[self.scalable_cols])
    
    def transform(self, X):
        if self.perform_scaling:
            if not hasattr(self, 'scalable_cols'):
                return RuntimeError(".fit() should be called first")
            
            X_transformed = X.copy()
            
            scaled_values = self.scaler.transform(X_transformed[self.scalable_cols])
            X_transformed[self.scalable_cols] = scaled_values.astype(float)
            
            return X_transformed
        
        return X