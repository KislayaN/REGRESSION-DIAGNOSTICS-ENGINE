import matplotlib.pyplot as plt

class Residual_plot:
    def __init__(self):
        pass
    
    def plot(self, y_test=None, y_pred=None, model_name=None):
        if y_test or y_pred is None:
            raise ValueError("y_test or y_pred can not be none")
        else: 
            if y_test and y_pred is None:
                raise ValueError("y_test or y_pred can not be none")
        if model_name is None:
            raise ValueError("model_name can not be none")
        
        residual = y_test - y_pred
        plt.scatter(y_pred, residual, alpha=0.5)
        plt.axhline(y=0, color='red', linestyle='--')
        plt.title(f'Residual Plot (Predicted vs Residuals) for {model_name}')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.grid(True, alpha=0.3)
        plt.show()