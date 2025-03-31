import numpy as np

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=1, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

class SoftmaxRegression():
    def __init__(self, *, W0=None, eta=0.01, eps=1e-4, max_iter=10000, penalty=None, c=1.0, l1_ratio=None, fit_intercept=True):
        self.W = W0
        self.eta = eta
        self.eps = eps
        self.max_iter = max_iter
        self.penalty = penalty
        self.c = c
        self.l1_ratio = l1_ratio
        self.fit_intercept= fit_intercept
                
    def fit(self, X, y):
        m, _ = X.shape
        cls = np.unique(y)
        k = len(cls)
        Y = np.eye(len(cls))[np.searchsorted(cls, y)]

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
 
        if self.W is None:
            self.W = np.zeros((k, X.shape[1]))

        def gradient():
            S = softmax(X @ self.W.T)
            G = -1 / m * (Y - S).T @ X
            if self.penalty:
                reg = (self.l1_ratio * np.sign(self.W) + (1 - self.l1_ratio) * self.W) / self.c
                reg[:, 0] = 0
                G += reg
            return G
        
        for _ in range(self.max_iter):
            G = gradient()
            W_new = self.W - self.eta * G

            if np.linalg.norm(W_new - self.W) < self.eps:
                break

            self.W = W_new

    def predict(self, X):
        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        S = X @ self.W.T
        return np.argmax(S, axis=1)