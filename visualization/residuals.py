import matplotlib.pyplot as plt

class Residual_plot:
    def __init__(self):
        pass
    
    def plot(self, y_test, y_pred):
        residual = y_test - y_pred
        plt.scatter(y_pred, residual, color='blue', alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title('Residual Plot (Predicted vs Residuals)')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        plt.show()