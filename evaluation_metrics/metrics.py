from sklearn.metrics import root_mean_squared_error, mean_squared_error, r2_score

class Get_metrics:
    def __init__(self):
        pass
    
    def rmse(self, y_true, y_pred):
        return root_mean_squared_error(y_true, y_pred)
    
    def mse(self, y_true, y_pred):
        return mean_squared_error(y_true, y_pred)
    
    def r2(self, y_true, y_pred):
        return r2_score(y_true, y_pred)
    
    def residual(self, y_true, y_pred):
        return y_true - y_pred