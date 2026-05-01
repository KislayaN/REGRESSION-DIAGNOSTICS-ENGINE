import seaborn as sns
import matplotlib.pyplot as plt

class Correlation_plot:
    def __init__(self):
        pass
    
    def plot(self, corr_matrix=None):
        "corr_matrix: the correlation matrix of dataframe (X and y included)"
        if corr_matrix is None:
            raise ValueError("corr_matrix can not be none")
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title("Correlation heatmap")
        plt.show()