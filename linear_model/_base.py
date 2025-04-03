import numpy as np


class LinearRegression:
    def __init__(self, fit_intercept=True):
        self.theta = None
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)


        self.theta = np.linalg.inv(X.T @ X) @ X.T @ y
        
    def predict(self, X):
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return X @ self.theta

    
