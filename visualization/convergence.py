import matplotlib.pyplot as plt

class Convergence_plot:
    def __init__(self, model_object, model_name):
        "model_name: name of model trained to get the convergence history"
        self.cost_history = None
        self.model_obj = model_object
        self.model_name = model_name
        
    def plot(self, convergence_history):
        self.cost_history = convergence_history
        
        plt.plot(self.cost_history, label=f"Convergence over {self.model_obj.n_iter} epochs for {self.model_name}")
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.xlim(0, self.model_obj.max_epochs)
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()