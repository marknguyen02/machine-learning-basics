import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

class Perceptron:
    def __init__(self,*, theta0=None, eta=0.01, max_iter=100, fit_intercept=True):
        self.theta = theta0
        self.eta = eta
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        m, _ = X.shape

        if self.fit_intercept:
            X = np.concatenate((np.ones((m, 1)), X), axis=1)
            np.concat

        if self.theta is None:
            self.theta = np.zeros(X.shape[1])
        
        for epoch in range(self.max_iter):
            has_err = False
            for i in range(m):
                y_pred = np.sign(self.theta @ X[i])
                if y_pred != y[i]:
                    has_err = True
                    self.theta += self.eta * (y[i] - y_pred) * X[i]
        
            if not has_err:
                break
        
    def predict(self, X):
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return np.sign(X @ self.theta)
