import numpy as np


def accuracy_score(y_true, y_pred, ndigits=2):
    _check_valid_length(y_true, y_pred)
    return round(np.mean(y_true == y_pred), ndigits)


def recall_score(y_true, y_pred, pos_label=1, ndigits=2):
    _check_valid_length(y_true, y_pred)
    m = len(y_true)

    tp = sum(y_pred[i] == pos_label == y_true[i] for i in range(m))
    fn = sum(y_pred[i] != pos_label == y_true[i] for i in range(m))

    if tp + fn == 0:
        return 0
    
    return round(tp / (tp + fn), ndigits)


def precision_score(y_true, y_pred, pos_label=1, ndigits=2):
    _check_valid_length(y_true, y_pred)
    m = len(y_true)

    tp = sum(y_pred[i] == pos_label == y_true[i] for i in range(m))
    fp = sum(y_pred[i] == pos_label != y_true[i] for i in range(m))

    if tp + fp == 0:
        return 0
    
    return round(tp / (tp + fp), ndigits)


def f1_score(y_true, y_pred, pos_label=1, ndigits=2):
    precision = precision_score(y_true, y_pred, pos_label, ndigits)
    recall = recall_score(y_true, y_pred, pos_label, ndigits)

    if precision + recall == 0:
        return 0
    
    result = 2 * precision * recall / (precision + recall)
    return round(result, ndigits)


def confusion_matrix(y_true, y_pred):
    _check_valid_length(y_true, y_pred)
    labels = np.sort(np.unique(y_true))
    n_labels = len(labels)
    m = len(y_true)
    cm = np.zeros((n_labels, n_labels), dtype=int)

    for i in range(n_labels):
        for j in range(n_labels):
            cm[i, j] = sum(y_true[k] == labels[i] and y_pred[k] == labels[j] for k in range(m))

    return cm


def classification_report(y_true, y_pred, ndigits=2):
    _check_valid_length(y_true, y_pred)
    labels = np.sort(np.unique(y_true))
    report = ''
    report += ' ' * 8
    report += 'precision'.rjust(10) + 'recall'.rjust(10) + 'f1-score'.rjust(10) + '\n\n'

    for label in labels:
        report += str(label).rjust(8)
        report += str(precision_score(y_true, y_pred, label, ndigits)).rjust(10)
        report += str(recall_score(y_true, y_pred, label, ndigits)).rjust(10)
        report += str(f1_score(y_true, y_pred, label, ndigits)).rjust(10)
        report += '\n'

    report += '\naccuracy' + ' ' * 10 * 2
    report += str(accuracy_score(y_true, y_pred, ndigits)).rjust(10)

    return report


def _check_valid_length(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError('y_true and y_pred must have the same length.')
