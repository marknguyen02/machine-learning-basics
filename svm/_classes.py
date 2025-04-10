import numpy as np
from metrics.pairwise import pairwise_kernel
from cvxopt import matrix, solvers


def solve_quadratic_programming(K, X, y, C):
    n = X.shape[0]
    P = matrix(np.outer(y, y) * K)
    q = matrix(-1.0 * np.ones(n))
    G_std = np.diag(-1.0 * np.ones(n))
    h_std = np.zeros(n)
    G_slack = np.identity(n)
    h_slack = np.ones(n) * C
    G = matrix(np.vstack((G_std, G_slack)))
    h = matrix(np.hstack((h_std, h_slack)))
    A = matrix(y.reshape(1, -1).astype(float))
    b = matrix(np.zeros(1))
    solvers.options['show_progress'] = False
    solutions = solvers.qp(P, q, G, h, A, b)
    return np.ravel(solutions['x'])


class SVC:
    def __init__(self, *, C=1, kernel='linear', tol=1e-6, degree=3, coef0=0.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.tol = tol
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.coef_ = None
        self.dual_coef_ = None
        self.intercept_ = None
        self.support_ = None
        self.support_vectors_ = None

                
    def fit(self, X, y):
        if self.gamma == 'scale':
            self.gamma = 1 / (X.shape[1] * X.var())
        elif self.gamma == 'auto':
            self.gamma = 1 / (X.shape[1])

        K = pairwise_kernel(
            X, 
            kernel=self.kernel,
            gamma=self.gamma,
            degree=self.degree,
            coef0=self.coef0
        )
        alpha = solve_quadratic_programming(K, X, y, self.C)
        self.support_ = np.where(alpha > self.tol)[0]        
        self.support_vectors_ = X[self.support_]
        margin_idx = (alpha[self.support_] < self.C - self.tol)
        if self.kernel == 'linear':
            self.coef_ = alpha[self.support_] * y[self.support_] @ self.support_vectors_
            self.intercept_ = np.mean(y[self.support_][margin_idx] - self.support_vectors_[margin_idx] @ self.coef_)
        else:
            self.dual_coef_ = alpha[self.support_] * y[self.support_]
            K_sv = K[self.support_][:, self.support_]
            self.intercept_ = np.mean(y[self.support_][margin_idx] - (K_sv @ self.dual_coef_)[margin_idx])

    def decision_function(self, X_test):
        if self.kernel == 'linear':
            return X_test @ self.coef_ + self.intercept_
        else:
            K_test = pairwise_kernel(
                self.support_vectors_,
                X_test,
                kernel=self.kernel,
                gamma=self.gamma, 
                degree=self.degree, 
                coef0=self.coef0
            )
            return self.dual_coef_ @ K_test + self.intercept_
           
    def predict(self, X_test):
        if self.kernel == 'linear':
            return np.sign(X_test @ self.coef_ + self.intercept_)
        else:
            K_test = pairwise_kernel(
                self.support_vectors_,
                X_test,
                kernel=self.kernel,
                gamma=self.gamma, 
                degree=self.degree, 
                coef0=self.coef0
            )
            return np.sign(self.dual_coef_ @ K_test + self.intercept_)