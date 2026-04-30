class Correlation:
    def __init__(self):
        self.corr = None
        
    def categorize_corr(self, c):
        c = abs(c)
        if c < 0.1:
            return 'negligible'
        elif c < 0.3:
            return 'weak'
        elif c < 0.5:
            return 'moderate'
        else: 
            return 'strong'
        
    def analyze_correlation(self, df, target):
        insights = []
        results = {}
        
        self.corr = df.corr()[target].drop(target)
        
        for feature, value in self.corr.items():
            strength = self.categorize_corr(value)
            results[feature] = {
                'Correlation': value,
                'Strength': strength
            }
            
            if strength == 'strong':
                insights.append(
                    f"{feature} has strong relationship with target"
                )
            
            return results, insights
                
    def compare_corr_vs_coeff(self, coeffs):
        insights = []
        corr = self.corr
        
        for index in corr.index:
            # Index is the feature
            if abs(corr[index]) > 0.5 and abs(coeffs[index]) < 0.01:
                insights.append(
                    f"{index} strongly correlated but low coefficient -> possible multicollinearity"
                )
                
        if insights == []:
            return "No insights found"
        
        return insights