from sklearn.impute import KNNImputer

class Imputer:
    def __init__(self):
        self.impute_method = KNNImputer(n_neighbors=5, weights='distance')
        self.transformed_data = None
        
    def fit(self, target):
        self.impute_method.fit(target)
        
    def transform(self, target):
        """The value you want to impute is target"""
        
        target = self.impute_method.transform(target)
        self.transformed_data = target
        return self.transformed_data