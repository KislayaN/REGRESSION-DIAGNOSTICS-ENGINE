import matplotlib.pyplot as plt

class Performance_plot:
    def __init__(self):
        self.perf_result = None
        
    def plot(self, performance_analysis_result=None):
        if performance_analysis_result is None:
            raise ValueError("performance_analysis_result can not be none")
        self.performance_result = performance_analysis_result
        models = self.performance_result.keys()
        mse_vals = self.performance_result.values()
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, mse_vals, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        
        for bar in bars:
            yval = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width()/2, # X-coordinate: center of the bar
                yval + 0.01,                     # Y-coordinate: just above the top
                f'{yval:.4f}',                   # The text (formatted to 4 decimal places)
                ha='center',                     # Horizontal alignment
                va='bottom',                     # Vertical alignment
                fontweight='bold'
            )

        plt.title('Model Comparison based on MSE', fontsize=14)
        plt.ylabel('Metric Value')
        plt.show()