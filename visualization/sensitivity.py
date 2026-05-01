import matplotlib.pyplot as plt

class Sensitivity_plot:
    def __init__(self):
        pass
    
    def plot_mse(self, mse_dict=None):
        if mse_dict is None:
            raise ValueError("mse_dict can not be none")
        
        self.iterator = iter(mse_dict.items())
        
        scaled_data_cost = next(self.iterator)
        unscaled_data_cost = next(self.iterator)
        
        print(scaled_data_cost)
        
        plt.plot(scaled_data_cost[1], label="Cost History for scaled data")
        plt.plot(unscaled_data_cost[1], label="Cost History for unscaled data")
        plt.legend()
        plt.ylabel("Coefficients value")
        plt.title("Convergence between MSE's of scaled and unscaled data")
        plt.yscale('log')
        plt.ylim(bottom=0.1)
        plt.show()