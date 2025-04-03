import numpy as np

class LogisticRegression:
    def __init__(self, *, theta0=None, lr=0.01, eps=1e-6, max_iter=10000, penalty=None, c=1.0, l1_ratio=None, fit_intercept=True):
        self.theta = theta0
        self.lr = lr
        self.max_iter = max_iter
        self.eps = eps
        self.penalty = penalty
        self.c = c
        self.l1_ratio = l1_ratio
        self.fit_intercept= fit_intercept

    def fit(self, X, y):
        m, _ = X.shape

        if self.penalty:
            if self.penalty == 'l1':
                self.l1_ratio = 1
            elif self.penalty == 'l2':
                self.l1_ratio = 0
            elif self.penalty == 'elasticnet':
                self.l1_ratio = 0.5
            else:
                raise ValueError(f"Invalid penalty type: {self.penalty}. Expected one of [None, 'l1', 'l2', 'elasticnet'].")
            
        if self.fit_intercept:
            X = np.concatenate((np.ones((m, 1)), X), axis=1)

        def h(theta):
            return 1 / (1 + np.exp(-theta @ X.T))
        
        def gradient(theta):
            grad = -1 / m * X.T @ (y - h(theta))
            if self.penalty:
                grad[1::] += (self.l1_ratio * np.sign(theta[1::]) + (1 - self.l1_ratio) * theta[1::]) / self.c
            return grad

        def next_theta(theta):
            return theta - self.lr * gradient(theta)
        
        if self.theta is None:
            self.theta = np.zeros(X.shape[1])
        
        old_theta = self.theta
        new_theta = next_theta(self.theta)
        for _ in range(self.max_iter):
            if np.linalg.norm(new_theta - old_theta, ord=2) < self.eps:
                break 
            old_theta = new_theta
            new_theta = next_theta(old_theta)


        self.theta = new_theta

    def predict(self, X):
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        return (self.theta @ X.T >= 0.5).astype(int)


