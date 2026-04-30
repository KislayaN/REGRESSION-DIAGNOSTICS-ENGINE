import numpy as np

from sklearn.preprocessing import StandardScaler

class Preprocessor:
    def __init__(self, perform_scaling=True):
        self.perform_scaling = perform_scaling
        self.scaler = StandardScaler()
        self.X_original = None
        
    def fit(self, X):
        self.scalable_cols = X.select_dtypes(include=[np.number]).columns
        self.scaler.fit(X[self.scalable_cols])
    
    def transform(self, X):
        self.X_original = X.copy()
        
        if self.perform_scaling:
            if not hasattr(self, 'scalable_cols'):
                return RuntimeError(".fit() should be called first")
            
            X_transformed = X.copy()
            
            scaled_values = self.scaler.transform(X_transformed[self.scalable_cols])
            X_transformed[self.scalable_cols] = scaled_values.astype(float)
            
            return X_transformed
        
        return X