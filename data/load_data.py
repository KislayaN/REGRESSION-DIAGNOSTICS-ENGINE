import pandas as pd

class Load_data:
    def __init__(self):
        self.data_loaded = False
        
    def load_data(self, data, target, path=None):
        if data is not None:
            dataframe = data.copy()
        elif path is not None:
            dataframe = pd.read_csv(path)
        else:
            raise ValueError("Provide data or path")
        
        if target not in dataframe:
            raise ValueError("Target not found in provided dataframe")
        
        self.dataframe = dataframe
        self.path = path
        self.target = target
        
        self.y = self.dataframe[target].values
        self.X = self.dataframe.drop(columns=[target]).values
        
        self.data_loaded = True