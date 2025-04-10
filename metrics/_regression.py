import numpy as np


def absolute_absolute_error(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred))


def mean_square_error(y_true, y_pred):
    return np.sum((y_true - y_pred) ** 2)


def r1_score(y_true, y_pred):
    pass