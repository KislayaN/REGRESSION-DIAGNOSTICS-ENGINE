import matplotlib.pyplot as plt

class Convergence_plot:
    def __init__(self, model_object, model_name):
        "model_name: name of model trained to get the convergence history"
        self.cost_history = None
        self.model_obj = model_object
        self.model_name = model_name
        
    def plot(self, convergence_history):
        self.cost_history = convergence_history
        
        plt.plot(convergence_history, label=f"Convergence over epochs {self.model_obj.n_iter} for {self.model_name}")
        plt.xlabel('Iterations')
        plt.xlim(self.model_name.n_iter)
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()
        
hist = Convergence_plot()