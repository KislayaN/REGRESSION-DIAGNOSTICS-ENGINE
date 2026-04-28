from sklearn.model_selection import train_test_split

class Splitter:
    def __init__(self):
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
    def split(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, random_state=43, test_size=0.2)
        
        return self.X_train, self.X_test, self.y_train, self.y_test