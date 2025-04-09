import numpy as np


class Perceptron:
    def __init__(self,*, theta0=None, lr=0.01, max_iter=100, fit_intercept=True):
        self.theta = theta0
        self.lr = lr
        self.max_iter = max_iter
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        m, _ = X.shape

        if self.fit_intercept:
            X = np.concatenate((np.ones((m, 1)), X), axis=1)
            np.concat

        if self.theta is None:
            self.theta = np.zeros(X.shape[1])
        
        for _ in range(self.max_iter):
            has_err = False
            for i in range(m):
                y_pred = np.sign(self.theta @ X[i])
                if y_pred != y[i]:
                    has_err = True
                    self.theta += self.lr * (y[i] - y_pred) * X[i]
        
            if not has_err:
                break
        
    def predict(self, X):
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return np.sign(X @ self.theta)
