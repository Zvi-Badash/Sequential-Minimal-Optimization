"""
this file implements a One Vs Rest multi class classifier.

"""

import numpy as np


class OneVsRestClassifier:
    def __init__(self, classifierClass, n_classes, class_labels=None, **kwargs):
        self.n_classes = n_classes
        self._classifiers = [classifierClass(**kwargs) for _ in range(self.n_classes)]
        if class_labels is not None:
            self.class_labels = class_labels
        else:
            self.class_labels = range(n_classes)

    def fit(self, X, y):
        for i, c in enumerate(self.class_labels):
            y_c = 1 * (y == c) - 1 * (y != c)
            self._classifiers[i].fit(X, y_c)

    def predict(self, x):
        idx = np.argmax([abs(self._classifiers[i].evaluate(x)) for i in range(self.n_classes)], axis=0)
        return self.class_labels[idx]

    def evaluate(self, x):
        return max([self._classifiers[i].evaluate(x) for i in range(self.n_classes)])
