import matplotlib.pyplot as plt

class Convergence_plot:
    def __init__(self):
        "model_name: name of model trained to get the convergence history"
        self.cost_history = None
        self.model_obj = None
        self.model_name = None
        
    def plot(self, convergence_history=None, model_name=''):
        if convergence_history is None:
            raise ValueError("convergence_history can not be none")
        self.model_name = model_name
        self.cost_history = convergence_history
        
        plt.plot(self.cost_history, label=f"Convergence over for {self.model_name}")
        plt.xlabel('Iterations')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(alpha=0.5)
        plt.show()