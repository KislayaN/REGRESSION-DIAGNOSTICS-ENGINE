import matplotlib.pyplot as plt

class Coefficient_plot:
    def __init__(self):
        pass
    
    def plot(self, coef_dict):
        """
        coef_dict: {
            ols: [values],
            lr: [values],
            lasso: [values],
            ridge: [values],
            gd_ols: [values]
        }
        """
        if len(coef_dict) != 5:
            raise ValueError("Exactly 5 Models are required for coefficients plot")
    
        plt.figure(figsize=(10, 6))
        
        for name, values in coef_dict.items():
            features = list(range(1, len(values) + 1))
            plt.plot(features, values, marker='o', mfc='white', ms=8, mew=1.5, label=name)

        plt.axhline(0, color='black', linewidth=1, alpha=0.5) # Reference line at 0
        
        plt.title("Coefficient Comparison Across Models", fontsize=14)
        plt.xlabel("Feature Index")
        plt.ylabel("Coefficient Value")
        
        plt.grid(axis='both', linestyle='--', alpha=0.3)
        plt.legend(loc='best')
        plt.tight_layout()
        plt.show()