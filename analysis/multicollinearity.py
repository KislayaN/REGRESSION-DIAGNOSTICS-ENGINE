class Multicollinearity:
    def __init__(self):
        self.insights = []
        
    def analyze_multicollinearity(self, corr_matrix, threshold=0.9):
        for i in range(len(corr_matrix)):
            for j in range(i):
                if abs(corr_matrix[i, j]) > threshold:
                    feat_1 = corr_matrix.columns[i]
                    feat_2 = corr_matrix.columns[j]
                    self.insights.append(f"High Collinearity between {feat_1} and {feat_2}")
        
        return self.insights