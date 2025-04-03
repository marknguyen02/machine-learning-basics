import numpy as np

class KNeighborsClassifier:
    def __init__(self, n_neighbors=10):
        self.k = n_neighbors
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        m, _ = X.shape
        y_pred = np.zeros(m)
        for i in range(m):
            distances = np.linalg.norm(self.X_train - X[i], axis=1)
            indices = np.argsort(distances)[:self.k]
            labels, counts = np.unique(self.y_train[indices], return_counts=True)
            y_pred[i] = labels[np.argmax(counts)]
        
        return y_pred
    

class KNeighborRegressor:
    def __init__(self, n_neighbors):
        self.k = n_neighbors

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        m, _ = X.shape
        y_pred = np.zeros(m)
        for i in range(m):
            distances = np.linalg.norm(self.X_train - X[i], axis=1)
            indices = np.argsort(distances)[:self.k]
            y_pred[i] = np.mean(self.y_train[indices], axis=1)
        
        return y_pred


