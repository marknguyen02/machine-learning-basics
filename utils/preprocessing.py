import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.var_ = None
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.var_ = np.var(X, axis=0)
        self.scale_ = np.sqrt(self.var_)
        return self
        
    def transform(self, X):
        X_scaled = (X - self.mean_) / self.scale_
        return X_scaled
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X):
        X_inverse = X * self.scale_ + self.mean_
        return X_inverse

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min_ = None
        self.scale_ = None
        self.data_min_ = None
        self.data_max_ = None
        self.data_range_ = None
        
    def fit(self, X):
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        min_val, max_val = self.feature_range
        self.scale_ = (max_val - min_val) / self.data_range_
        self.min_ = min_val - self.data_min_ * self.scale_
        return self
        
    def transform(self, X):
        X_scaled = X * self.scale_ + self.min_
        return X_scaled
        
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
        
    def inverse_transform(self, X):
        X_inverse = (X - self.min_) / self.scale_
        return X_inverse