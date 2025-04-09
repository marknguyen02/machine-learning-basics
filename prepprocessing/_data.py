import numpy as np
import pandas as pd


class MinMaxScaler:
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit_transform(self, X):
        self.min_val = np.min(X, axis=0)
        self.max_val = np.max(X, axis=0)
        return (X - self.min_val) / (self.max_val - self.min_val)
    
    def transform(self, X):
        if self.min_val is None or self.max_val is None:
            raise ValueError('')
        return (X - self.min_val) / (self.max_val - self.min_val)
    

class StandardScaler:
    def __init__(self):
        self.u = None
        self.s = None

    def fit_transform(self, X):
        self.u = np.mean(X, axis=0)
        self.s = np.std(X, axis=0)
        return (X - self.u) / self.s
    
    def transform(self, X):
        if self.u is None or self.s is None:
            raise ValueError('')
        return (X - self.u) / self.s