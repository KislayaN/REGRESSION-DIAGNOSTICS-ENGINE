from sklearn.model_selection import train_test_split

class Splitter:
    def __init__(self):
        self.splitter = train_test_split()
        
    def split(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = self.splitter(X, y, random_state=43, test_size=0.2)
        
        return self.X_train, self.X_test, self.y_train, self.y_test