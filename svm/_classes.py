import numpy as np
from metrics.pairwise import pairwise_kernel
from cvxopt import matrix, solvers


def solve_svc_qp(K, y, C):
    n = y.shape[0]
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
        alpha = solve_svc_qp(K, y, self.C)
        self.support_ = np.where(alpha > self.tol)[0]        
        self.support_vectors_ = X[self.support_]
        margin = np.where((alpha > self.tol) & (alpha < self.C - self.tol))[0]        
        if self.kernel == 'linear':
            self.coef_ = alpha[self.support_] * y[self.support_] @ self.support_vectors_
            self.intercept_ = np.mean(y[margin] - X[margin] @ self.coef_)
        else:
            self.dual_coef_ = alpha[self.support_] * y[self.support_]
            self.intercept_ = np.mean(y[margin] - (K @ (alpha * y))[margin])

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
            return K_test @ self.dual_coef_ + self.intercept_
           
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
            return np.sign(K_test @ self.dual_coef_ + self.intercept_)
        

def solve_svr_qp(K, y, C):
    n = y.shape[0]
    K_combined = np.vstack((np.hstack((K, -K)), np.hstack((-K, K))))
    P = matrix(K_combined)
    epsilon = 0.1
    q = matrix(np.hstack([epsilon - y, epsilon + y]))
    G_std = np.vstack([-np.eye(2 * n), np.eye(2 * n)])
    h_std = np.hstack([np.zeros(2 * n), np.ones(2 * n) * C])
    G = matrix(G_std)
    h = matrix(h_std)
    A = matrix(np.hstack([np.ones(n), -np.ones(n)]), (1, 2 * n))
    b = matrix(0.0)
    solvers.options['show_progress'] = False
    solution = np.ravel(solvers.qp(P, q, G, h, A, b)['x'])
    alpha = solution[:n]
    alp_star = solution[n:]
    return alpha, alp_star


class SVR:
    def __init__(self, *, C=1, kernel='linear', tol=1e-4, degree=3, coef0=0.0, gamma='scale'):
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
        alpha, alp_star = solve_svr_qp(K, y, self.C)
        self.support_ = np.where((alpha > self.tol) | (alp_star > self.tol))[0]
        self.support_vectors_ = X[self.support_]
        margin = np.where(((alpha > self.tol) & (alpha < self.C - self.tol)) | ((alp_star > self.tol) & (alp_star < self.C - self.tol)))[0]

        if self.kernel == 'linear':
            self.coef_ = (alpha - alp_star)[self.support_] @ self.support_vectors_
            self.intercept_ = np.mean(y[margin] - X[margin] @ self.coef_)
        else:
            self.dual_coef_ = (alpha - alp_star)[self.support_]
            self.intercept_ = np.mean(y[margin] - (K @ (alpha - alp_star))[margin])


    def predict(self, X_test):
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