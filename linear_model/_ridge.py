import numpy as np


class Ridge:
    def __init__(self, *, w=None, epsilon=1e-4, ld=1, max_iter=1000, lr=0.01, fit_intercept=True):
        self.w = w
        self.epsilon = epsilon
        self.alpha = 1
        self.max_iter = max_iter
        self.lr = lr
        self.fit_intercept = fit_intercept
    
    def fit(self, X, y):
        m, _ = X.shape

        if self.fit_intercept:
            X = np.concatenate((np.ones((m, 1)), X), axis=1)
        
        
    
        



    
    def predict(self, X):
        m, _ = X.shape
        if self.fit_intercept:
            X = np.concatenate((np.ones((m, 1)), X), axis=1)
        return X @ self.w