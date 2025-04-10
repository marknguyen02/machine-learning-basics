import numpy as np


def _kernel(x1, x2, *, kernel, gamma, degree, coef0):
    if kernel == 'linear':
        return np.dot(x1, x2)
    elif kernel == 'rbf':
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
    elif kernel == 'poly':
        return (gamma * np.dot(x1, x2) + coef0) ** degree
    else: 
        raise ValueError(f"Unsupported kernel type: '{kernel}'.")  


def pairwise_kernel(X1, X2=None, *, kernel='rbf', gamma, degree, coef0):
    if X2 is None:
        X2 = X1
    m = X1.shape[0]
    n = X2.shape[0]
    K = np.empty((m, n))
    
    for i in range(m):
        for j in range(n):
            K[i][j] = _kernel(X1[i, :], X2[j, :], kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)
    return K


def linear_kernel(X):
    return pairwise_kernel(X, kernel='linear')


def polynomial_kernel(X, *, degree, coef0, gamma):
    return pairwise_kernel(X, kernel='poly', degree=degree, coef0=coef0, gamma=gamma)


def rbf_kernel(X, *, gamma):
    return pairwise_kernel(X, kernel='rbf', gamma=gamma)