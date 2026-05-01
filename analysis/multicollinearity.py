import numpy as np
import pandas as pd

class Multicollinearity:
    def __init__(self):
        self.insights = []
        
    def analyze_multicollinearity(self, corr_matrix, threshold=0.9):
        corr_matrix = abs(corr_matrix)
        mask = np.triu(np.ones_like(corr_matrix, dtype='bool'), k = 1)
        upper = corr_matrix.where(mask)
        
        pairs = upper.stack()
        
        strong_pairs = pairs[pairs > threshold]
        if strong_pairs.empty:
            raise ValueError("Try reducing threhold, there are no features above this threshold")
        
        pairs_index = [idx for idx in strong_pairs.index[0]]
        self.insights.append(f"{pairs_index[0]} and {pairs_index[1]} are highly correlated with each other")
        
        return self.insights
        