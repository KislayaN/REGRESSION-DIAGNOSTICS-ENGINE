class Sensitivity:
    def __init__(self):
        self.insights = []
        self.strong_features_scaled = []
        self.strong_features_unscaled = []
        
    def sort_dict(self, dict):
        return sorted(dict.items(), key=lambda x: abs(x[1]), reverse=True)
    
    def check_sensitivity(self, mse_dict=None, coefficients_dict=None, top_features=None):
        """
        mse_dict = {
            'mse_unscaled_data': 0.0,
            'mse_scaled_data': 0.0
        }
        
        coefficients_dict = {
            'unscaled': {
                'feature-1': value,
                'feature-2': value, etc.
            }
            'scaled': {
                'feature-1': value,
                'feature-2': value, etc. 
            }
        }
        """
        
        if len(mse_dict) != 2:
            raise ValueError("Only two values are accepted in mse dict")
        
        if len(coefficients_dict) != 2:
            raise ValueError("Only two values are accepted in coefficients dict")
        
        mse_unscaled_data = list(mse_dict.values())[0]
        mse_scaled_data = list(mse_dict.values())[1]

        coef_unscaled_data = list(coefficients_dict.values())[0]
        coef_scaled_data = list(coefficients_dict.values())[1]

        if mse_unscaled_data is None:
            raise ValueError("The given mse for unscaled data is none")
        if mse_scaled_data is None:
            raise ValueError("The given mse for scaled data is none")
        
        if coef_unscaled_data is None:
            raise ValueError("The given coefficients for unscaled data is none")
        if coef_scaled_data is None:
            raise ValueError("The given coefficients for scaled data is none")
        
        if mse_scaled_data < mse_unscaled_data:
            self.insights.append("Scaling reduced error")
        
        coef_unscaled_data = list(coefficients_dict.items())[0][1]
        coef_scaled_data = list(coefficients_dict.items())[1][1]
        
        if top_features == 0 or top_features is None:
            return []

        scaled_sorted_coef = self.sort_dict(coef_scaled_data)
        unscaled_sorted_coef = self.sort_dict(coef_unscaled_data)
        
        for feat in range(top_features):
            self.strong_features_scaled.append(scaled_sorted_coef[feat])
            self.strong_features_unscaled.append(unscaled_sorted_coef[feat])
            
        return self.strong_features_scaled, self.strong_features_unscaled, self.insights